import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor
from scipy.constants import pi
from scipy.constants import kilo

import numpy as np


class SOLProperties(om.Group):
    r"""Outputs properties of the SOL and divertor

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

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
        Power fraction to outer divertor
    θ_pol : float
        Poloidal tilt at plate
    θ_tot : float
        Total B-field angle of incidence at strike point
    f_fluxexp : float
        Poloidal flux expansion factor

    Notes
    -----
    The f_rad specified here is independent of the calculation actually used
    for the core plasma. In the future these should be coupled together.
    """
    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

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

        acc.set_output(ivc,
                       f,
                       "poloidal tilt at plate",
                       component_name="θ_pol",
                       units="rad")
        acc.set_output(ivc,
                       f,
                       "total B field incidence angle",
                       component_name="θ_tot",
                       units="rad")
        acc.set_output(ivc,
                       f,
                       "poloidal flux expansion",
                       component_name="f_fluxexp")
        acc.set_output(ivc, f, "SOL width multiplier")
        self.add_subsystem("ivc", ivc, promotes=["*"])


class StrikePointRadius(om.ExplicitComponent):
    r"""Radius of the critical strike point

    Three models are implemented. For the snowflake divertor (SF),

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

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Inputs
    ------
    R0 : float
       m, Plasma geometric major radius
    a : float
       m, Plasma minor radius
    δ : float
       Triangularity

    Outputs
    -------
    R_strike : float
       m, Outer strike point major radius

    Notes
    -----
    The choice of model is configured using
    ``plasma:SOL:divertor:model``. The options are either ``"SF"``,
    ``"SXD"``, or ``"LinearDelta"``.

    The configurations for each component are within
    ``plasma:SOL:divertor:[Model choice]``.

    For the SF model, the shift factor :math:`c` is loaded via
    ``outer strike point radius shift (SF, SFD)``.

    For the Super-X model, :math:`c` is loaded via
    ``outer strike point radius multiplier``.

    For the LinearDelta model, :math:`f_w` and :math:`f_d` are loaded via
    ``outer-inner width`` and ``delta factor``.


    Raises
    ------
    ValueError
       If the model string is not recognized.
    """
    BAD_MODEL = "Only 'SF', 'SXD', and 'LinearDelta' are supported"

    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("StrikePointRadius requries a config file")
        self.add_input("R0", units="m", desc="Plasma geometric major radius")
        self.add_input("a", units="m", desc="Minor radius")
        self.add_input("δ", val=0, desc="Triangularity")
        config = self.options["config"].accessor(["plasma", "SOL", "divertor"])
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
        self.add_output("R_strike",
                        units="m",
                        lower=0,
                        ref=R_strike_ref,
                        desc="Outer strike point major radius")

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


class GoldstonHDSOL(om.ExplicitComponent):
    r"""SOL width and seperatrix temperature

    The scrape-off-layer width and upstream seperatrix temperature
    are computed according Equations (6) and (7) of the
    Heuristic Drift ("HD") model :footcite:p:`goldston_heuristic_2012`.

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
        MA, Plasma current
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
        mm, SOL width, assuming that ion magnetic drift determines the net
        particle transport
    """
    def setup(self):
        self.c_Tesep = 30.81
        self.c_λ = 5671
        self.add_input("R0", units="m", desc="Plasma geometric major radius")
        self.add_input("a", units="m", desc="Minor radius")
        self.add_input("κ", desc="Elongation")
        self.add_input("Bt", units="T", desc="Vacuum toroidal field on axis")
        self.add_input("Ip", units="A", desc="Plasma current")
        self.add_input("P_sol", units="W", desc="Power into SOL")
        self.add_input("Z_eff", desc="Effective ion charge")
        self.add_input("Z-bar", desc="Average ion charge")
        self.add_input("A-bar", desc="Average ion mass")

        T_sep_ref = 100
        self.add_output("T_sep",
                        units="eV",
                        lower=0,
                        ref=T_sep_ref,
                        desc="Midplane seperatrix temperature")
        λ_ref = 1e-3
        self.add_output("λ", units="m", lower=0, ref=λ_ref, desc="SOL width")

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

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Notes
    -----
    Which model is used is configured via
    ``plasma:SOL:divertor:peak heat flux model``
    and the choices are either ``"poloidal angle"`` or ``"total angle"``.

    Inputs
    ------
    R_strike : float
        m, Strike point radius
    κ : float
        Plasma elongation
    P_sol : float
        MW, Power to SOL
    f_outer : float
        Fraction of power to the outer divertor
    f_fluxexp : float
        Factor by which power is spread out due to flux expansion.
        (Typically greater than 1.)
    λ_sol : float
        m, SOL width
    q_star : float
        Cylindrical safety factor
    θ_pol : float
        rad, Poloidal angle of the field at the strike point
    θ_tot : float
        rad, Total angle of the field to the strike point surface

    N_div : float
        Number of divertors (1 or 2). 1 is for single-null, 2 is a double-null.

    Outputs
    -------
    q_max : float
        MW/m**2, Peak heat flux

    Raises
    ------
    ValueError
        If the model choice string is not recognized.
    """
    POLOIDAL_ANGLE_MODEL = 1
    TOTAL_ANGLE_MODEL = 2
    BAD_MODEL = "Only 'poloidal angle' and 'total angle' are supported"

    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("PeakHeatFlux requries a config file")

        config = self.options["config"].accessor(["plasma", "SOL", "divertor"])
        model = config(["peak heat flux model"])
        if model == "poloidal angle":
            self.model = self.POLOIDAL_ANGLE_MODEL
        elif model == "total angle":
            self.model = self.TOTAL_ANGLE_MODEL
        else:
            raise ValueError(self.BAD_MODEL)

        self.add_input("R_strike", units="m", desc="Strike point major radius")
        self.add_input("κ", desc="Elongation")
        self.add_input("P_sol", units="MW", desc="Power to SOL")
        self.add_input("f_outer",
                       desc="Fraction of power to the outer divertor")
        self.add_input("f_fluxexp", desc="Flux expansion factor")
        self.add_input("λ_sol", units="mm", desc="SOL width")
        self.add_input("q_star", desc="Cylindrical safety factor")
        self.add_input("θ_pol",
                       units="rad",
                       desc="Poloidal angle of the field at strike point")
        self.add_input(
            "θ_tot",
            units="rad",
            desc="Total angle of the field to the strike point surface")
        self.add_input("N_div", 1, desc="Number of divertors")
        q_max_ref = 3e6
        self.add_output("q_max",
                        units="MW/m**2",
                        lower=0,
                        ref=q_max_ref,
                        desc="Peak heat flux")

    def compute(self, inputs, outputs):
        Rs = inputs["R_strike"]
        κ = inputs["κ"]
        P_sol = inputs["P_sol"]
        f_out = inputs["f_outer"]
        f_fluxexp = inputs["f_fluxexp"]
        λ_sol = inputs["λ_sol"] / kilo
        q_star = inputs["q_star"]
        θ_pol = inputs["θ_pol"]
        θ_tot = inputs["θ_tot"]
        N_div = inputs["N_div"]

        if self.model == self.POLOIDAL_ANGLE_MODEL:
            q_max = P_sol * f_out * np.sin(θ_pol) / (N_div * 2 * pi * Rs *
                                                     f_fluxexp * λ_sol)
            outputs["q_max"] = q_max
        elif self.model == self.TOTAL_ANGLE_MODEL:
            q_max = P_sol * q_star * np.sin(θ_tot) * f_out / (N_div * 2 * pi *
                                                              Rs * κ * λ_sol)
            outputs["q_max"] = q_max
        else:
            raise ValueError(self.BAD_MODEL)

    def setup_partials(self):
        self.declare_partials("q_max",
                              ["R_strike", "P_sol", "f_outer", "λ_sol"],
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
    r"""Top-level tokamak SOL and divertor group

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Inputs
    ------
    R0 : float
       m, Plasma major radius
    a : float
       m, Plasma minor radius
    κ : float
        Plasma elongation
    δ : float
       Triangularity
    Bt : float
        T, Vacuum toroidal field on-axis
    Ip : float
        MA, Plasma current
    q_star : float
        Normalized safety factor
    P_heat : float
        MW, Plasma heating power

    Outputs
    -------
    P_sol : float
        MW, Power entering the SOL
    T_sep : float
        eV, Midplane seperatrix temperature
    λ_HD : float
        mm, SOL width, assuming that ion magnetic drift determines the net
        particle transport
    λ_sol : float
        mm, adjusted SOL width
    q_max : float
        MW/m**2, Peak heat flux

    Notes
    -----
    The T_sep computation is an output, and is not guaranteed to be consistent
    with any model of the core.
    """
    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           SOLProperties(config=config),
                           promotes_outputs=["*"])
        self.add_subsystem(
            "Psol",
            om.ExecComp(
                "P_sol = P_heat * (1 - f_rad)",
                P_sol={
                    'units': "MW",
                    'lower': 0.1,
                    'desc': "Power entering the SOL"
                },
                f_rad={'desc': "Fraction of heating power which is radiated"},
                P_heat={
                    "units": "MW",
                    'desc': "Plasma heating power"
                }),
            promotes=["*"])
        self.add_subsystem("R_strike",
                           StrikePointRadius(config=config),
                           promotes_inputs=["*"],
                           promotes_outputs=["R_strike"])
        self.add_subsystem("HD",
                           GoldstonHDSOL(),
                           promotes_inputs=["*"],
                           promotes_outputs=["T_sep", ("λ", "λ_HD")])
        self.add_subsystem(
            "lambda_q",
            om.ExecComp("lambda_q = lambda_q_HD * fudge_factor",
                        lambda_q={
                            "units": "mm",
                            'desc': "Heat flux width"
                        },
                        fudge_factor={'desc': "Adjustment factor"},
                        lambda_q_HD={
                            "units": "mm",
                            'desc': "Computed heat flux width"
                        }),
            promotes_inputs=[("lambda_q_HD", "λ_HD"),
                             ("fudge_factor", "SOL width multiplier")],
            promotes_outputs=[("lambda_q", "λ_sol")])
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
    prob.set_val("Bt", 2.094, units="T")
    prob.set_val("Ip", 14.67, units="MA")
    prob.set_val("P_heat", 119.06, units="MW")
    prob.set_val("q_star", 3.56)

    prob.run_driver()

    all_inputs = prob.model.list_inputs(val=True,
                                        desc=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(val=True,
                                          desc=True,
                                          print_arrays=True,
                                          units=True)
