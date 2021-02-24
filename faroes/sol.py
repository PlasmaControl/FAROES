import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor
from scipy.constants import pi
from scipy.constants import kilo

import numpy as np


class SOLProperties(om.ExplicitComponent):
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
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"].accessor(["plasma", "SOL"])
        n_div = config(["number of divertors"])
        self.add_discrete_output("N_div", val=n_div)

        acc = Accessor(self.options["config"])
        f = acc.accessor(["plasma", "SOL", "plasma mix"])
        acc.set_output(self, f, "Z_eff")
        acc.set_output(self, f, "Z-bar")
        acc.set_output(self, f, "A-bar")
        f = acc.accessor(["plasma", "SOL"])
        acc.set_output(self,
                       f,
                       "core radiated power fraction",
                       component_name="f_rad")
        acc.set_output(self,
                       f,
                       "power fraction to outer divertor",
                       component_name="f_outer")

        acc.set_output(self,
                       f,
                       "poloidal tilt at plate",
                       component_name="θ_pol",
                       units="rad")
        acc.set_output(self,
                       f,
                       "total B field incidence angle",
                       component_name="θ_tot",
                       units="rad")
        acc.set_output(self,
                       f,
                       "poloidal flux expansion",
                       component_name="f_fluxexp")
        acc.set_output(self, f, "SOL width multiplier")


class StrikePointRadius(om.ExplicitComponent):
    r"""Radius of the critical strike point

    This is computed in one of two ways. For the snowflake divertor (SF),

    .. math::

       R_\mathrm{strike} = R_0 - c a

    where c is a constant,

    and for the super-X divertor (SXD),

    .. math::

       R_\mathrm{strike} = c R_0

    where c is a (different) constant.

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

    Outputs
    -------
    R_strike : float
       m, Outer strike point major radius
    """
    BAD_MODEL = "Only 'SF' or 'SXD' are supported"

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("StrikePointRadius requries a config file")
        self.add_input("R0", units="m")
        self.add_input("a", units="m")
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
        else:
            raise ValueError(self.BAD_MODEL)

        R_strike_ref = 5
        self.add_output("R_strike", units="m", lower=0, ref=R_strike_ref)

    def compute(self, inputs, outputs):
        R0 = inputs["R0"]
        a = inputs["a"]

        if self.model == "SF":
            c = self.c_sf
            outputs["R_strike"] = R0 - c * a
        elif self.model == "SXD":
            c = self.c_sxd
            outputs["R_strike"] = c * R0
        else:
            raise ValueError(self.BAD_MODEL)

    def setup_partials(self):
        if self.model == "SF":
            c = self.c_sf
            self.declare_partials("R_strike", ["R0"], val=1)
            self.declare_partials("R_strike", ["a"], val=-c)
        elif self.model == "SXD":
            c = self.c_sxd
            self.declare_partials("R_strike", ["R0"], val=c)
            self.declare_partials("R_strike", ["a"], dependent=False)
        else:
            raise ValueError(self.BAD_MODEL)


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
    :code:`plasma:SOL:divertor:peak heat flux model`
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
    λ_sol : float
        m, SOL width
    q_star : float
        Normalized safety factor
    θ_pol : float
        rad, Poloidal angle of the field at the strike point
    θ_tot : float
        rad, Total angle of the field to the strike point surface

    Discrete inputs
    ---------------
    N_div : int
        Number of divertors (1 or 2). 1 is for single-null, 2 is a double-null.

    Outputs
    -------
    q_max : float
        MW/m**2, Peak heat flux
    """
    POLOIDAL_ANGLE_MODEL = 1
    TOTAL_ANGLE_MODEL = 2
    BAD_MODEL = "Only 'poloidal angle' and 'total angle' are supported"

    def initialize(self):
        self.options.declare('config', default=None)

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

        self.add_input("R_strike", units="m")
        self.add_input("κ")
        self.add_input("P_sol", units="MW")
        self.add_input("f_outer")
        self.add_input("f_fluxexp")
        self.add_input("λ_sol", units="mm")
        self.add_input("q_star")
        self.add_input("θ_pol", units="rad")
        self.add_input("θ_tot", units="rad")
        self.add_discrete_input("N_div", 1)
        q_max_ref = 3e6
        self.add_output("q_max", units="MW/m**2", lower=0, ref=q_max_ref)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        Rs = inputs["R_strike"]
        κ = inputs["κ"]
        P_sol = inputs["P_sol"]
        f_out = inputs["f_outer"]
        f_fluxexp = inputs["f_fluxexp"]
        λ_sol = inputs["λ_sol"] / kilo
        q_star = inputs["q_star"]
        θ_pol = inputs["θ_pol"]
        θ_tot = inputs["θ_tot"]
        N_div = discrete_inputs["N_div"]

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
                                       P_sol={"units": "MW"},
                                       P_heat={"units": "MW"}),
                           promotes=["*"])
        self.add_subsystem("R_strike",
                           StrikePointRadius(config=config),
                           promotes_inputs=["*"],
                           promotes_outputs=["R_strike"])
        self.add_subsystem("HD",
                           GoldstonHDSOL(),
                           promotes_inputs=["*"],
                           promotes_outputs=["T_sep", ("λ", "λ_HD")])
        self.add_subsystem("lambda_q",
                           om.ExecComp("lambda_q = lambda_q_HD * fudge_factor",
                                       lambda_q={"units": "mm"},
                                       lambda_q_HD={"units": "mm"}),
                           promotes_inputs=[("lambda_q_HD", "λ_HD"),
                                            ("fudge_factor",
                                             "SOL width multiplier")],
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

    all_inputs = prob.model.list_inputs(values=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=True,
                                          units=True)
