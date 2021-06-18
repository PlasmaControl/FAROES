import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor
from scipy.constants import pi
from scipy.constants import kilo
from faroes.util import PowerScalingLaw

import numpy as np


class SOLProperties(om.Group):
    r"""
    Outputs
    -------
    Z_eff : float
        SOL effective ion charge
    Z-bar : float
        SOL average ion charge
    A-bar : float
        u, SOL average ion mass
    f_rad : float
        Core radiated power fraction
    f_outer : float
        power fraction to outer divertor
    θ_pol : float
        poloidal tilt at plate
    θ_tot : float
        Total B-field angle of incidence at strike point
    f_fluxexp : float
        Poloidal flux expansion factor
    """
    POLOIDAL_ANGLE_MODEL = "poloidal angle"
    TOTAL_ANGLE_MODEL = "total angle"
    BAD_MODEL = "Only 'poloidal angle' and 'total angle' are supported"

    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        ivc = om.IndepVarComp()
        config = self.options["config"].accessor(["plasma", "SOL"])
        n_div = config(["number of divertors"])
        ivc.add_output("N_div", val=n_div)

        acc = Accessor(self.options["config"])
        f = acc.accessor(["plasma", "SOL", "plasma mix"])
        acc.set_output(ivc, f, "Z_eff")
        acc.set_output(ivc, f, "Z-bar")
        acc.set_output(ivc, f, "A-bar")
        f = acc.accessor(["plasma", "SOL"])
        acc.set_output(ivc,
                       f,
                       "core radiated power fraction",
                       component_name="f_rad")
        acc.set_output(ivc,
                       f,
                       "power fraction to outer divertor",
                       component_name="f_outer")

        acc.set_output(ivc, f, "SOL width multiplier")

        loc = ["plasma", "SOL", "peak heat flux"]
        config = self.options["config"].accessor(loc)
        model = config(["model"])

        if model == self.POLOIDAL_ANGLE_MODEL:
            f = acc.accessor(loc + [model])
            acc.set_output(ivc,
                           f,
                           "poloidal tilt at plate",
                           component_name="θ_pol",
                           units="rad")
            acc.set_output(ivc,
                           f,
                           "poloidal flux expansion",
                           component_name="f_fluxexp")
        elif model == self.TOTAL_ANGLE_MODEL:
            f = acc.accessor(loc + [model])
            acc.set_output(ivc,
                           f,
                           "total B field incidence angle",
                           component_name="θ_tot",
                           units="rad")
        else:
            raise ValueError(self.BAD_MODEL)
        self.add_subsystem("ivc", ivc, promotes=["*"])


class StrikePointRadius(om.ExplicitComponent):
    r"""Radius of the critical strike point

    This is computed in one of three ways. For the snowflake divertor (SF),

    .. math::

       R_\mathrm{strike} = R_0 - c a

    where c is a constant,

    and for the super-X divertor (SXD),

    .. math::

       R_\mathrm{strike} = c R_0

    where c is a (different) constant.

    For the LinearDelta model,

    .. math::

       R_\mathrm{strike} = R_0 + f_w a - f_d δ a,

    where typically :math:`f_w = 1/4` and `f_d = 1`.

    Notes
    -----
    This component is configured using
    config:SOL:divertor:model. The options are either "SF" or "SXD".

    Inputs
    ------
    R0 : float
       m, Plasma major radius
    a : float
       m, Plasma minor radius
    δ : float
       Triangularity

    Outputs
    -------
    R_strike : float
       m, Outer strike point major radius
    """
    BAD_MODEL = "Only 'SF', 'SXD', and 'LinearDelta' are supported"

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("StrikePointRadius requries a config file")
        self.add_input("R0", units="m")
        self.add_input("a", units="m")
        self.add_input("δ", val=0)
        config = self.options["config"].accessor(
            ["plasma", "SOL", "divertor geometry"])
        model = config(["model"])
        self.model = model
        if model == "SFD" or model == "SF":
            self.model = "SF"
            self.c_sf = config(
                ["SF", "outer strike point radius shift (SF, SFD)"])
        elif model == "SXD":
            self.c_sxd = config(
                ["SXD", "outer strike point radius multiplier"])
        elif model == "LinearDelta":
            self.fw = config(["LinearDelta", "outer-inner width"])
            self.fd = config(["LinearDelta", "delta factor"])
        else:
            raise ValueError(self.BAD_MODEL)

        R_strike_ref = 5
        self.add_output("R_strike", units="m", lower=0, ref=R_strike_ref)

    def compute(self, inputs, outputs):
        R0 = inputs["R0"]
        a = inputs["a"]
        δ = inputs["δ"]

        if self.model == "SF":
            c = self.c_sf
            outputs["R_strike"] = R0 - c * a
        elif self.model == "SXD":
            c = self.c_sxd
            outputs["R_strike"] = c * R0
        elif self.model == "LinearDelta":
            fw = self.fw
            fd = self.fd
            outputs["R_strike"] = R0 + fw * a - fd * a * δ
        else:
            raise ValueError(self.BAD_MODEL)

    def setup_partials(self):
        if self.model == "SF":
            c = self.c_sf
            self.declare_partials("R_strike", ["R0"], val=1)
            self.declare_partials("R_strike", ["a"], val=-c)
            self.declare_partials("R_strike", ["δ"], dependent=False)
        elif self.model == "SXD":
            c = self.c_sxd
            self.declare_partials("R_strike", ["R0"], val=c)
            self.declare_partials("R_strike", ["a"], dependent=False)
            self.declare_partials("R_strike", ["δ"], dependent=False)
        elif self.model == "LinearDelta":
            self.declare_partials("R_strike", ["R0"], val=1)
            self.declare_partials("R_strike", ["a"])
            self.declare_partials("R_strike", ["δ"])
        else:
            raise ValueError(self.BAD_MODEL)

    def compute_partials(self, inputs, J):
        if self.model == "LinearDelta":
            a = inputs["a"]
            δ = inputs["δ"]
            fw = self.fw
            fd = self.fd
            J["R_strike", "a"] = fw - fd * δ
            J["R_strike", "δ"] = -fd * a
        else:
            pass


class HoracekSOLWidth(om.Group):
    r"""The outer midplane heat flux width

    .. math ::

       \lambda_q^\mathrm{omp} / \mathrm{mm} = C_0
          \left(R_0/\mathrm{m}\right)^{C_{R}}
          \left(\epsilon\right)^{C_{\epsilon}}
          \left(S_c/\mathrm{m}^2\right)^{C_{S_c}}
          \left(S_{LCFS}/\mathrm{m}^2\right)^{C_{S_{LCFS}}}
          \left(B_\phi/\mathrm{T}\right)^{C_{B_{\phi}}}
          \left(I_p/\mathrm{MA}\right)^{C_{I_p}}
          \left(q_{95}\right)^{C_{q95}}
          \left(q_\mathrm{cyl}\right)^{C_{q_{cyl}}}
          \left(P_\mathrm{SOL}/\mathrm{MW}\right)^{C_{P_{SOL}}}
          \left(f_\mathrm{Gw}\right)^{C_{f_{Gw}}}
          \left(\frac{\left<p\right>\right}{\mathrm{atm})^{C_{\left<p\right>}}

    This Component constructs a power law equation based on a set of exponents
    from a configuration file. Not all the possible inputs (which depend on the
    choice of scaling law) will be constructed.

    Inputs
    ------
    R0 : float
        m, major radius.
    ε : float
        Inverse aspect ratio.
    S_c : float
        m**2, Cross sectional area of the plasma.
    S_LCFS : float
        m**2, Outer surface area of LCFS.
    Bφ : float
        T, vacuum toroidal field.
    Ip : float
        MA, Plasma current.
    B_pol : float
        T, Poloidal field at the outer midplane
    q95 : float
        Near-edge safety factor.
    q_cyl : float
        Cylindrical safety factor. Also called q*.
    P_sol : float
        MW, Non-radiation power through the LCFS.
    f_Gw: float
        Greenwald fraction.
    <p> : float
        atm, average plasma pressure.

    Outputs
    -------
    λq : float
        mm, Outer midplane heat flux decay length

    References
    ----------
    .. [1] Horacek, J. et al.
       Scaling of L-Mode Heat Flux for ITER and COMPASS-U Divertors,
       Based on Five Tokamaks.
       Nucl. Fusion 2020, 60 (6), 066016.
       https://doi.org/10.1088/1741-4326/ab7e47.

    Notes
    -----
    Developer notes: this class structure is somewhat copied from
    the confinement time law. The two could be generalized and united.
    """

    CONST = "c0"
    BAD_TERM = """Unknown term '%s' in SOL power width scaling.
    Valid terms are %s """
    NEGATIVE_TERM = "Term '%s' is non-positive in " + \
        "_the confinement time calculation. Its value was %f."

    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        self.OUTPUT = "λq"

        config = self.options["config"]
        model_acc = config.accessor(["plasma", "SOL"])
        scaling = model_acc(["heat flux decay length model"])

        fit_acc = config.accessor(["fits", "heat flux decay length"])
        terms = fit_acc([scaling]).copy()

        valid_terms = {
            self.CONST: "mm",
            "R0": "m",
            "ε": None,
            "S_c": "m**2",
            "S_LCFS": "m**2",
            "Bφ": "T",
            "Ip": "A",
            "B_pol": "T",
            "q95": None,
            "q_cyl": None,
            "P_sol": "MW",
            "f_Gw": None,
            "<p>": "atm",
        }

        for k, v in terms.items():
            if k not in valid_terms.keys():
                raise ValueError(self.BAD_TERM % (k, valid_terms))

        self.add_subsystem("law",
                           PowerScalingLaw(terms=terms,
                                           term_units=valid_terms,
                                           const=self.CONST,
                                           output=self.OUTPUT,
                                           lower=1e-3),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])

        # Send terms that are not in the scaling law to an 'ignore' function

        # some terms must be renamed because greek, etc, is not compabile with
        # ExecComps
        renames = {"ε": "epsilon", "<p>": "p_avg"}
        ignore_terms = {}
        ignore_equation = "ignore = 0.0"
        promotes_list = []
        for k, v in valid_terms.items():
            if k != self.CONST and k not in terms.keys():
                kp = k
                if k in renames.keys():
                    kp = renames[k]
                    promotes_list += [(kp, k)]
                ignore_terms[kp] = {"units": valid_terms[k]}
                ignore_equation += " + 0.0 * " + kp

        promotes_list.append("*")

        if len(ignore_terms) > 0:
            self.add_subsystem("ignore",
                               om.ExecComp([ignore_equation], **ignore_terms),
                               promotes_inputs=promotes_list)


class SpreadingWidthEstimate(om.ExplicitComponent):
    r"""Integral heat flux spreading width

    This is a rough estimate for the heat flux spreading.
    I just take it as half of λq.
    """
    def setup(self):
        self.add_input("λq", units='mm')
        self.add_output("S", val=0, units='mm')
        self.const = 0.5

    def compute(self, inputs, outputs):
        outputs["S"] = self.const * inputs["λq"]

    def setup_partials(self):
        self.declare_partials("S", "λq", val=self.const)


class LambdaInt(om.ExplicitComponent):
    r"""The integral SOL power flux width

    This quantity is in units of width-at-the-midplane
    and is inversely proportional to peak heat fluxes on the target.

    .. math::

       \lambda_\mathrm{int} = \lambda_q + 1.64 S

    Inputs
    ------
    λq : float
        mm, Power flux width at the midplane
    S : float
        mm, Width of a gaussian spreading parameter
    Inputs
    ------
    λint : float
        mm, 'Integral' power flux width at midplane

    Notes
    -----
    This is an approximation for the peak heat flux of the Eich fitting
    function [2]_. It is derived in the appendix of [1]_. Eich's work
    addresses H-mode SOL physics.

    References
    ----------
    .. [1] Makowski, M. A., et al.
       Analysis of a Multi-Machine Database on Divertor Heat Fluxes.
       Physics of Plasmas 2012, 19 (5), 056122.
       https://doi.org/10.1063/1.4710517.

    .. [2] Eich, T.; Sieglin, B.; Scarabosio, A.; Fundamenski, W.;
       Goldston, R. J.; Herrmann, A.; ASDEX Upgrade Team.
       Inter-ELM Power Decay Length for JET and ASDEX Upgrade:
       Measurement and Comparison with Heuristic Drift-Based Model.
       Physical Review Letters 2011, 107 (21).
       https://doi.org/10.1103/PhysRevLett.107.215001.
    """
    def setup(self):
        self.add_input("λq", units="mm")
        self.add_input("S", val=0.0, units="mm")
        self.add_output("λint", units="mm", lower=1e-4)
        self.const = 1.64

    def compute(self, inputs, outputs):
        λq = inputs["λq"]
        S = inputs["S"]
        outputs["λint"] = λq + self.const * S

    def setup_partials(self):
        c = self.const
        self.declare_partials("λint", "λq", val=1)
        self.declare_partials("λint", "S", val=c)


class GoldstonHDSOL(om.ExplicitComponent):
    r"""

    The scrape-off-layer width and upstream seperatrix temperature
    are computed according the Heuristic Drift ("HD") model [1]_.

    .. math::

       \lambda/\mathrm{mm} =
           5.761 \left(\frac{P_\mathrm{SOL}}{\mathrm{W}}\right)^{1/8}
           (1 + \kappa^2)^{5/8}
           \left(\frac{a}{\mathrm{m}}\right)^{17/8}
           \left(\frac{B_T }{ \mathrm{T}}\right)^{1/4} \\
           \left(\frac{I_p }{ \mathrm{A}}\right)^{-9/8}
           \left(\frac{R}{\mathrm{m}}\right)^{-1}
           \left(\frac{2 \bar{A}}{\bar{Z}^2(1 + \bar{Z})}\right)^{7/16}
           \left(\frac{Z_\mathrm{eff} + 4}{5}\right)^{1/8}

    .. math::

       T_\mathrm{sep}/\mathrm{eV} =
           30.81 \left(\frac{P_\mathrm{SOL}}{\mathrm{W}}\right)^{1/4}
           \left(\frac{1 + \bar{Z}}{2 \bar{A}}\right)^{1/8}
           \left(\frac{a}{\mathrm{m}}\right)^{1/4} \\
           \left(1+\kappa^2\right)^{1/4}
           \left(\frac{B_T}{\mathrm{T}}\right)^{1/2}
           \left(\frac{I_p}{\mathrm{A}}\right)^{-1/4}
           \left(\frac{Z_\mathrm{eff} + 4}{5}\right)^{1/4}

    Inputs
    ------
    R0 : float
        m, Plasma major radius
    a : float
        m, Plasma minor radius
    κ : float
        Plasma elongation.
    Bt : float
        T, Vacuum toroidal field on-axis
    Ip : float
        A, Plasma current
    P_sol : float
        W, Power into SOL
    Z_eff : float
        Effective ion charge.
    Z-bar : float
        Average ion charge.
    A-bar : float
        u, Average ion mass.

    Outputs
    -------
    T_sep : float
        eV, Midplane seperatrix temperature
    λ : float
        mm, midplane integral SOL width,
        assuming that ion magnetic drift determines the net particle transport

    References
    ----------
    .. [1] Goldston, R. J.
       Heuristic Drift-Based Model of the Power Scrape-off Width
       in Low-Gas-Puff H-Mode Tokamaks.  Nuclear Fusion 2012, 52, 013009.
       https://doi.org/10.1088/0029-5515/52/1/013009.
       Equations (6) and (7).
    """
    def setup(self):
        # reference: Menard T109
        self.c_Tesep = 30.81
        self.c_λ = 5671
        self.add_input("R0", units="m")
        self.add_input("a", units="m")
        self.add_input("κ")
        self.add_input("Bt", units="T")
        self.add_input("Ip", units="A")
        self.add_input("P_sol", units="W")
        self.add_input("Z_eff")
        self.add_input("Z-bar")
        self.add_input("A-bar")

        T_sep_ref = 100
        self.add_output("T_sep", units="eV", lower=0, ref=T_sep_ref)
        λ_ref = 1e-3
        self.add_output("λ", units="m", lower=0, ref=λ_ref)

    def compute(self, inputs, outputs):
        R0 = inputs["R0"]
        a = inputs["a"]
        κ = inputs["κ"]
        Bt = inputs["Bt"]
        Ip = inputs["Ip"]

        if Ip <= 0:
            raise om.AnalysisError(f"GoldstonHDSolWidth: Ip = {Ip} < 0")

        P_sol = inputs["P_sol"]
        Z_eff = inputs["Z_eff"]
        Z_bar = inputs["Z-bar"]
        A_bar = inputs["A-bar"]

        T_sep = self.c_Tesep * (P_sol**(1 / 4) *
                                (Z_bar**2 * (1 + Z_bar) /
                                 (2 * A_bar))**(1 / 8) * a**(1 / 4) *
                                (1 + κ**2)**(1 / 4) * Bt**(1 / 2) *
                                Ip**(-1 / 4) * ((Z_eff + 4) / 5)**(1 / 4))

        λ = self.c_λ * (P_sol**(1 / 8) * (1 + κ**2)**(5 / 8) * a**(17 / 8) *
                        Bt**(1 / 4) * Ip**(-9 / 8) * R0**(-1) *
                        ((Z_eff + 4) / 5)**(1 / 8) *
                        (2 * A_bar / (Z_bar**2 * (1 + Z_bar)))**(7 / 16))
        outputs["T_sep"] = T_sep
        outputs["λ"] = λ

    def setup_partials(self):
        self.declare_partials(
            "T_sep",
            ["P_sol", "Z-bar", "A-bar", "Z_eff", "a", "κ", "Bt", "Ip"],
            method="cs")
        self.declare_partials(
            "λ",
            ["P_sol", "κ", "a", "Bt", "Ip", "R0", "A-bar", "Z-bar", "Z_eff"],
            method="cs")


class PeakHeatFlux(om.ExplicitComponent):
    r"""Peak heat flux at strike point

    Poloidal angle model:

    .. math::

       q_{max} = P_\mathrm{SOL} \frac{f_\mathrm{outer}}
                 {N_\mathrm{div} 2 \pi R_\mathrm{strike}
                 \lambda_\mathrm{SOL}} \frac{\sin(\theta_\mathrm{pol})}{
                 f_\mathrm{flux exp}}

    Total angle model:

    .. math::

       q_{max} = P_\mathrm{SOL} \frac{f_\mathrm{outer}}
                 {N_\mathrm{div} 2 \pi R_\mathrm{strike}
                 \lambda_\mathrm{SOL}} \frac{q^*}{\kappa}
                 \sin(\theta_\mathrm{tot})

    Notes
    -----
    Which model is used is configured via
    :code:`plasma:SOL:divertor:peak heat flux:model`
    and the choices are either :code:`poloidal angle` or :code:`total angle`.

    Inputs
    ------
    R_strike : float
        m, Strike point radius
    κ : float
        Plasma elongation
    P_sol : float
        MW, Power to SOL.
    f_outer : float
        Fraction of power to the outer divertor.
    f_fluxexp : float
        Factor by which power is spread out due to flux expansion.
        (Typically greater than 1.)
    λ_int : float
        m, Integral SOL width
    q_star : float
        Normalized safety factor
    θ_pol : float
        rad, Poloidal angle of the field at the strike point
    θ_tot : float
        rad, Total angle of the field to the strike point surface

    N_div : int
        Number of divertors (1 or 2). 1 is for single-null, 2 is a double-null.

    Outputs
    -------
    q_max : float
        MW/m**2, Peak heat flux
    """
    POLOIDAL_ANGLE_MODEL = "poloidal angle"
    TOTAL_ANGLE_MODEL = "total angle"
    MODELS = [POLOIDAL_ANGLE_MODEL, TOTAL_ANGLE_MODEL]
    BAD_MODEL = "Only 'poloidal angle' and 'total angle' are supported"

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("PeakHeatFlux requries a config file")

        config = self.options["config"].accessor(["plasma", "SOL"])
        model = config(["peak heat flux", "model"])
        if not any(model == m for m in self.MODELS):
            raise ValueError(self.BAD_MODEL)
        else:
            self.model = model

        self.add_input("R_strike", units="m")
        self.add_input("κ")
        self.add_input("P_sol", units="MW")
        self.add_input("f_outer")
        self.add_input("f_fluxexp")
        self.add_input("λ_int", units="mm")
        self.add_input("q_star")
        self.add_input("θ_pol", units="rad")
        self.add_input("θ_tot", units="rad")
        self.add_input("N_div", 1)
        q_max_ref = 3e6
        self.add_output("q_max", units="MW/m**2", lower=0, ref=q_max_ref)

    def compute(self, inputs, outputs):
        Rs = inputs["R_strike"]
        κ = inputs["κ"]
        P_sol = inputs["P_sol"]
        f_out = inputs["f_outer"]
        f_fluxexp = inputs["f_fluxexp"]
        λ_int = inputs["λ_int"] / kilo
        q_star = inputs["q_star"]
        θ_pol = inputs["θ_pol"]
        θ_tot = inputs["θ_tot"]
        N_div = inputs["N_div"]

        if self.model == self.POLOIDAL_ANGLE_MODEL:
            q_max = P_sol * f_out * np.sin(θ_pol) / (N_div * 2 * pi * Rs *
                                                     f_fluxexp * λ_int)
            outputs["q_max"] = q_max
        elif self.model == self.TOTAL_ANGLE_MODEL:
            q_max = P_sol * q_star * np.sin(θ_tot) * f_out / (N_div * 2 * pi *
                                                              Rs * κ * λ_int)
            outputs["q_max"] = q_max
        else:
            raise ValueError(self.BAD_MODEL)

    def setup_partials(self):
        self.declare_partials("q_max",
                              ["R_strike", "P_sol", "f_outer", "λ_int"],
                              method="cs")
        self.declare_partials("q_max", ["N_div"], method="cs")
        if self.model == self.POLOIDAL_ANGLE_MODEL:
            self.declare_partials("q_max", ["f_fluxexp", "θ_pol"], method="cs")
            self.declare_partials("q_max", ["q_star", "θ_tot", "κ"],
                                  dependent=False)
        elif self.model == self.TOTAL_ANGLE_MODEL:
            self.declare_partials("q_max", ["θ_tot", "q_star", "κ"],
                                  method="cs")
            self.declare_partials("q_max", ["f_fluxexp", "θ_pol"],
                                  dependent=False)
        else:
            raise ValueError(self.BAD_MODEL)


class SOLAndDivertor(om.Group):
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           SOLProperties(config=config),
                           promotes_outputs=["*"])
        self.add_subsystem("Psol",
                           om.ExecComp("P_sol = P_heat * (1 - f_rad)",
                                       P_sol={
                                           "units": "MW",
                                           "lower": 0.1
                                       },
                                       P_heat={"units": "MW"}),
                           promotes=["*"])
        self.add_subsystem("R_strike",
                           StrikePointRadius(config=config),
                           promotes_inputs=["*"],
                           promotes_outputs=["R_strike"])

        model_accessor = config.accessor(["plasma", "SOL"])
        model = model_accessor(["heat flux decay length model"])
        if model == "Goldston HD":
            self.add_subsystem("power_sol_width",
                               GoldstonHDSOL(),
                               promotes_inputs=["*"],
                               promotes_outputs=["T_sep", ("λ", "λq")])

            # make an 'ignore' subsystem to catch unused variables
            ignore_eq = "ignore = 0 * (eps + p_avg + f_Gw + B_pol + q95 + S_c)"
            self.add_subsystem("ignore",
                               om.ExecComp([ignore_eq],
                                           p_avg={'units': 'atm'},
                                           B_pol={'units': 'T'},
                                           S_c={'units': 'm**2'}),
                               promotes_inputs=[("eps", "ε"), ("p_avg", "<p>"),
                                                "*"])

        else:
            self.add_subsystem("power_sol_width",
                               HoracekSOLWidth(config=config),
                               promotes_inputs=[("Bφ", "Bt"),
                                                ("q_cyl", "q_star"), "*"],
                               promotes_outputs=["λq"])
            # For consistency with Menard,
            # the spreading of the Goldston HD model is 0
            self.add_subsystem("spreading",
                               SpreadingWidthEstimate(),
                               promotes_inputs=["λq"],
                               promotes_outputs=["S"])

        self.add_subsystem("integral_sol_width",
                           LambdaInt(),
                           promotes_inputs=["λq", "S"],
                           promotes_outputs=["λint"])

        self.add_subsystem("lambda_int",
                           om.ExecComp("lambda_out = lambda_in * fudge_factor",
                                       lambda_in={"units": "mm"},
                                       lambda_out={"units": "mm"}),
                           promotes_inputs=[("lambda_in", "λint"),
                                            ("fudge_factor",
                                             "SOL width multiplier")],
                           promotes_outputs=[("lambda_out", "λ_int")])
        self.add_subsystem("peak_heat_flux",
                           PeakHeatFlux(config=config),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])
        self.set_input_defaults("P_sol", units="MW", val=10)


if __name__ == "__main__":
    uc = UserConfigurator()
    prob = om.Problem()
    prob.model = SOLAndDivertor(config=uc)
    prob.setup()

    prob.set_val("κ", 2.74)
    prob.set_val("R0", 3.0, units="m")
    prob.set_val("a", 1.875, units="m")
    prob.set_val("ε", 0.625)
    prob.set_val("f_Gw", 0.8)
    prob.set_val("Bt", 2.094, units="T")
    prob.set_val("Ip", 14.67, units="MA")
    prob.set_val("P_heat", 119.06, units="MW")
    prob.set_val("q_star", 3.56)

    # ITER numbers for validation with Horacek
    prob.set_val("Bt", 5.3, units="T")
    prob.set_val("B_pol", 1.14, units="T")
    prob.set_val("<p>", 0.86, units='atm')
    prob.set_val("q95", 1.7)
    prob.set_val("q_star", 1.6)
    prob.set_val("ε", 0.28)
    prob.set_val("Ip", 12.00, units="MA")
    prob.set_val("S_c", 13.8, units="m**2")
    prob.set_val("f_Gw", 0.33)
    prob.set_val("R0", 6.5, units="m")
    prob.set_val("P_heat", 90.0, units="MW")

    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=True,
                                          units=True)
