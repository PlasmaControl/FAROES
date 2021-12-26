import openmdao.api as om

from scipy.constants import pi
from scipy.special import jv
from scipy.integrate import quad
from faroes.shapefactor import ConstProfile, ParabProfileConstTriang
from faroes.shapefactor import ParabProfileLinearTriang


class PedestalProfileConstTriang(om.ExplicitComponent):
    r"""Model for pedestal profiles and constant triangularity

    Inputs
    ------
    A : float
        None, Aspect ratio (R0 / a0)
    δ0 : float
        None, Triangularity of border curve of plasma distribution
    κ : float
        None, Elongation of plasma distribution shape
    αn : float
        None, exponent in density profile (peaking parameter)
    αT : float
        None, exponent in temperature profile (peaking parameters)
    β : float
        None, second exponent in temperature profile (chosen freely by user)
    ρpedn : float
        None, value of normalized radius at density pedestal top
    ρpedT : float
        None, value of normalized radius at temperature pedestal top
    n0 : float
        m**(-3), density at center
    nped : float
        m**(-3), density at pedestal top
    n1 : float
        m**(-3), density at separatix
    T0 : float
        keV, temperature at center
    Tped : float
        keV, temperature at pedestal top
    T1 : float
        keV, temperature at separatix


    Outputs
    ------
    S : float
        m**(-6) * keV**(1 / 2), shape factor


    Notes
    ------
    OpenMDAO does not currently support fractional exponents for units,
    so the units for the shape factor have been set as W / m**(-3). This
    is to effectively include the necessary units from the constant in the
    power calculation, c, that will result in the units W / m**(-3) for S.

    """

    def setup(self):
        self.add_input("A", desc="major radius")
        self.add_input("δ0", desc="border triangularity")
        self.add_input("κ", desc="elongation")

        self.add_input("αn", desc="density peaking parameter")
        self.add_input("αT", desc="temperature peaking parameter")
        self.add_input("β", val=2., desc="second exponent for temperature")

        self.add_input("ρpedn", val=1., desc="density barrier")
        self.add_input("ρpedT", val=1., desc="temperature barrier")

        self.add_input("n0", units="m**(-3)", desc="center dens")
        self.add_input("nped", units="m**(-3)", val=0., desc="ped top dens")
        self.add_input("n1", units="m**(-3)", val=0., desc="separatix dens")

        self.add_input("T0", units="keV", desc="center temp")
        self.add_input("Tped", units="keV", val=0., desc="ped top temp")
        self.add_input("T1", units="keV", val=0., desc="separatix temp")

        self.add_output("S", units="W * m**(-3)", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]

        αn = inputs["αn"]
        αT = inputs["αT"]
        β = inputs["β"]

        ρpedn = inputs["ρpedn"]
        ρpedT = inputs["ρpedT"]

        n0 = inputs["n0"]
        nped = inputs["nped"]
        n1 = inputs["n1"]

        T0 = inputs["T0"]
        Tped = inputs["Tped"]
        T1 = inputs["T1"]

        def integrand(ρ):
            term1 = 4 * A * (jv(0, δ0) + jv(2, δ0))
            term2 = 3 / 2 * ρ * (jv(1, 2 * δ0) + jv(3, 2 * δ0))
            dVdρ = pi**2 * κ * ρ * (term1 - term2)

            if 0 <= ρ <= ρpedn:
                dens = nped + (n0 - nped) * (1 - ρ**2 / (ρpedn**2))**αn
            else:
                # if ρpedn < ρ <= 1
                dens = n1 + (nped - n1) * (1 - ρ)/(1 - ρpedn)

            if 0 <= ρ <= ρpedT:
                temp = Tped + (T0 - Tped) * (1 - ρ**β / (ρpedT**β))**αT
            else:
                # if ρpedT < ρ <= 1
                temp = T1 + (Tped - T1) * (1 - ρ)/(1 - ρpedT)

            return dVdρ * dens**2 * temp**(1 / 2)

        outputs["S"] = quad(integrand, 0, 1)[0]

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ", "αn", "αT",
                                    "β", "ρpedn", "ρpedT", "n0",
                                    "nped", "n1", "T0", "Tped", "T1"])


class PedestalProfileLinearTriang(om.ExplicitComponent):
    r"""Model for pedestal profiles and constant triangularity

    Inputs
    ------
    A : float
        None, Aspect ratio (R0 / a0)
    δ0 : float
        None, Triangularity of border curve of plasma distribution
    κ : float
        None, Elongation of plasma distribution shape
    αn : float
        None, exponent in density profile (peaking parameter)
    αT : float
        None, exponent in temperature profile (peaking parameters)
    β : float
        None, second exponent in temperature profile (chosen freely by user)
    ρpedn : float
        None, value of normalized radius at density pedestal top
    ρpedT : float
        None, value of normalized radius at temperature pedestal top
    n0 : float
        m**(-3), density at center
    nped : float
        m**(-3), density at pedestal top
    n1 : float
        m**(-3), density at separatix
    T0 : float
        keV, temperature at center
    Tped : float
        keV, temperature at pedestal top
    T1 : float
        keV, temperature at separatix


    Outputs
    ------
    S : float
        m**(-6) * keV**(1/2), shape factor
        (add Notes about incorporating keV**(1/2) into this from constant c)


    Notes
    ------
    OpenMDAO does not currently support fractional exponents for units,
    so the units for the shape factor have been set as W / m**(-3). This
    is to effectively include the necessary units from the constant in the
    power calculation, c, that will result in the units W / m**(-3) for S.

    """

    def setup(self):
        self.add_input("A", desc="major radius")
        self.add_input("δ0", desc="border triangularity")
        self.add_input("κ", desc="elongation")

        self.add_input("αn", desc="density peaking parameter")
        self.add_input("αT", desc="temperature peaking parameter")
        self.add_input("β", val=2., desc="second exponent for temperature")

        self.add_input("ρpedn", val=1., desc="density barrier")
        self.add_input("ρpedT", val=1., desc="temperature barrier")

        self.add_input("n0", units="m**(-3)", val=1., desc="center density")
        self.add_input("nped", units="m**(-3)", val=0., desc="ped top density")
        self.add_input("n1", units="m**(-3)", val=0., desc="separatix density")

        self.add_input("T0", units="keV", val=1., desc="center temperature")
        self.add_input("Tped", units="keV", val=0., desc="ped top temperature")
        self.add_input("T1", units="keV", val=0., desc="separatix temperature")

        self.add_output("S", units="W * m**(-3)", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]

        αn = inputs["αn"]
        αT = inputs["αT"]
        β = inputs["β"]

        ρpedn = inputs["ρpedn"]
        ρpedT = inputs["ρpedT"]

        n0 = inputs["n0"]
        nped = inputs["nped"]
        n1 = inputs["n1"]

        T0 = inputs["T0"]
        Tped = inputs["Tped"]
        T1 = inputs["T1"]

        def integrand(ρ):
            term1 = 4 * A * ρ * (jv(0, δ0 * ρ) + jv(2, δ0 * ρ))
            term2 = -2 * δ0 * ρ**3 * (jv(0, 2 * δ0 * ρ) + jv(2, 2 * δ0 * ρ))
            term3 = -A * δ0 * ρ**2 * (jv(1, δ0 * ρ) + jv(3, δ0 * ρ))
            dVdρ = pi**2 * κ * (term1 + term2 + term3)

            if 0 <= ρ <= ρpedn:
                dens = nped + (n0 - nped) * (1 - ρ**2 / (ρpedn**2))**αn
            elif ρpedn < ρ <= 1:
                dens = n1 + (nped - n1) * (1 - ρ)/(1 - ρpedn)

            if 0 <= ρ <= ρpedT:
                temp = Tped + (T0 - Tped) * (1 - ρ**β / (ρpedT**β))**αT
            elif ρpedT < ρ <= 1:
                temp = T1 + (Tped - T1) * (1 - ρ)/(1 - ρpedT)

            return dVdρ * dens**2 * temp**(1 / 2)

        outputs["S"] = quad(integrand, 0, 1)[0]

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ", "αn", "αT",
                                    "β", "ρpedn", "ρpedT", "n0",
                                    "nped", "n1", "T0", "Tped", "T1"])


class Bremsstrahlung(om.Group):
    r"""

    Computes the Bremsstrahlung radiation for a plasma distribution based on
    the profile and triangularity functions.

    Notes
    ------
    The constant self.c is derived using the nonrelativstic Born approximation
    for Bremsstrahlung radiation as described in Johner [1]_. We have that

    .. math::

       c = \frac{32e^6 \sqrt{2k}}{3(4 \pi \epsilon_0)^3
           m_e^{3/2} c^3 \hbar \sqrt{\pi}} = 5.355 \cdot 10^{-37}

    References
    ----------
    .. [1] Johner Jean (2011) HELIOS: A Zero-Dimensional Tool for Next Step and
       Reactor Studies, Fusion Science and Technology, 59:2, 308-349,
       DOI: 10.13182/FST11-A11650

    """

    def initialize(self):
        self.options.declare("config", default=None, recordable=False)
        self.options.declare("profile", default="constant")
        self.options.declare("triangularity", default="constant")

    def setup(self):
        profile = self.options["profile"]
        triangularity = self.options["triangularity"]

        self.c = 5.355e-37

        if profile == "constant":
            self.add_subsystem("const_profile",
                               ConstProfile(),
                               promotes_inputs=["A", "δ0", "κ"],
                               promotes_outputs=["S"])

            brems_eq = "P=S * a0**3 * Zeff**2 * n0**2 * T0**(1/2) * "
            self.add_subsystem("brems",
                               om.ExecComp(brems_eq + str(self.c),
                                           a0={"units": "m"},
                                           n0={"units": "m**(-3)"},
                                           T0={"units": "keV"},
                                           P={"units": "W"}),
                               promotes=["*"])

            ignore_eq1 = "ignore = 0 * (alpha + alphan + alphaT + beta + "
            ignore_eq2 = "rhopedn + rhopedT + nped + n1 + Tped + T1)"
            inputs_to_promote = [("alpha", "α"), ("alphan", "αn"),
                                 ("alphaT", "αT"), ("beta", "β"),
                                 ("rhopedn", "ρpedn"), ("rhopedT", "ρpedT"),
                                 "nped", "n1", "Tped", "T1"]
            self.add_subsystem("ignore",
                               om.ExecComp([ignore_eq1 + ignore_eq2],
                                           nped={"units": "m**(-3)"},
                                           n1={"units": "m**(-3)"},
                                           Tped={"units": "keV"},
                                           T1={"units": "keV"}),
                               promotes_inputs=inputs_to_promote)

        elif profile == "parabolic":
            if triangularity == "constant":
                self.add_subsystem("parab_profile_const_triang",
                                   ParabProfileConstTriang(),
                                   promotes_inputs=["A", "δ0", "κ", "α"],
                                   promotes_outputs=["S"])
            elif triangularity == "linear":
                self.add_subsystem("parab_profile_linear_triang",
                                   ParabProfileLinearTriang(),
                                   promotes_inputs=["A", "δ0", "κ", "α"],
                                   promotes_outputs=["S"])

            brems_eq = "P=S * a0**3 * Zeff**2 * n0**2 * T0**(1/2) * "
            self.add_subsystem("brems",
                               om.ExecComp(brems_eq + str(self.c),
                                           a0={"units": "m"},
                                           n0={"units": "m**(-3)"},
                                           T0={"units": "keV"},
                                           P={"units": "W"}),
                               promotes=["*"])

            ignore_eq1 = "ignore = 0 * (alphan + alphaT + beta + "
            ignore_eq2 = "rhopedn + rhopedT + nped + n1 + Tped + T1)"
            inputs_to_promote = [("alphan", "αn"), ("alphaT", "αT"),
                                 ("beta", "β"), ("rhopedn", "ρpedn"),
                                 ("rhopedT", "ρpedT"),
                                 "nped", "n1", "Tped", "T1"]
            self.add_subsystem("ignore",
                               om.ExecComp([ignore_eq1 + ignore_eq2],
                                           nped={"units": "m**(-3)"},
                                           n1={"units": "m**(-3)"},
                                           Tped={"units": "keV"},
                                           T1={"units": "keV"}),
                               promotes_inputs=inputs_to_promote)

        elif profile == "pedestal":
            if triangularity == "constant":
                self.add_subsystem("pedestal_profile_const_triang",
                                   PedestalProfileConstTriang(),
                                   promotes_inputs=["A", "δ0", "κ",
                                                    "αn", "αT", "β",
                                                    "ρpedn", "ρpedT", "n0",
                                                    "nped", "n1", "T0",
                                                    "Tped", "T1"],
                                   promotes_outputs=["S"])
            elif triangularity == "linear":
                self.add_subsystem("pedestal_profile_linear_triang",
                                   PedestalProfileLinearTriang(),
                                   promotes_inputs=["A", "δ0", "κ",
                                                    "αn", "αT", "β",
                                                    "ρpedn", "ρpedT", "n0",
                                                    "nped", "n1", "T0",
                                                    "Tped", "T1"],
                                   promotes_outputs=["S"])
            brems_eq = "P=S * a0**3 * Zeff**2 * "
            self.add_subsystem("brems",
                               om.ExecComp(brems_eq + str(self.c),
                                           a0={"units": "m"},
                                           S={"units": "W * m**(-3)"},
                                           P={"units": "W"}),
                               promotes=["*"])

            self.add_subsystem("ignore",
                               om.ExecComp("ignore=0 * alpha"),
                               promotes_inputs=[("alpha", "α")])


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = Bremsstrahlung(profile="pedestal", triangularity="constant")

    prob.setup(force_alloc_complex=True)
    prob.set_val("A", 5 / 2)
    prob.set_val("δ0", 0.2)
    prob.set_val("κ", 1)
    prob.set_val("α", 2)

    prob.set_val("a0", 5)
    prob.set_val("Zeff", 4)
    prob.set_val("n0", 2e20)
    prob.set_val("T0", 2, units="keV")

    prob.set_val("αn", 0.8)
    prob.set_val("αT", 0.8)
    prob.set_val("β", 2)

    prob.set_val("ρpedn", 0.5)
    prob.set_val("ρpedT", 0.9)

    prob.set_val("nped", 2)
    prob.set_val("n1", 5)
    prob.set_val("Tped", 2)
    prob.set_val("T1", 5)

    prob.run_driver()
    all_inputs = prob.model.list_inputs(val=True)
    all_outputs = prob.model.list_outputs(val=True)
