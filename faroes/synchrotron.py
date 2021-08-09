from faroes.sauter_plasma import SauterGeometry
import faroes.units  # noqa: F401

import openmdao.api as om
from scipy.constants import pi
import numpy as np


class SynchrotronFit(om.ExplicitComponent):
    r"""Estimated power for Synchrotron radiation

    Calculates synchrotron radiation from fit from Albajar [1]_ (p. 674),
    given by:

    .. math::
       P_{syn, r} = &3.84 \times 10^{-8}(1-r)^{1/2} Ra^{1.38}\kappa^{0.79} \\
           &\times B_t^{2.62}n_0^{0.38}T_0(16+T_0)^{2.61} \\
           &\times \left(1+0.12\frac{T_0}{p_a^{0.41}}\right)^{-1.51} \\
           &\times K(\alpha_n, \alpha_T, \beta_T) G(A)

    where K and G are profile and aspect ratio factors, respectively

    .. math::
       K(\alpha_n, \alpha_T, \beta_T) =&(\alpha_n+3.87\alpha_T+1.46)^{-0.79}\\
           &\times (1.98 + \alpha_T)^{1.36}\beta_T^{2.14}\\
           &\times (\beta_T^{1.53}+1.87\alpha_T-0.16)^{-1.33},\\

    .. math::
       G(A) = 0.93[1+0.85\text{exp}(-0.82A)].

    Inputs
    ------
    A : float
        Aspect ratio (R0 / a0)
    a0 : float
        m, minor radius
    κ : float
        Elongation of plasma distribution

    αn : float
        Exponent in density profile (peaking parameter)
    αT : float
        Exponent in temperature profile (peaking parameter)
    βT : float
        Exponent for ρ in temperature profile (peaking parameter)

    Bt : float
        T, toroidal magnetic field
    pa : float
        Optical thickness of the plasma
    r : float
        Wall reflectivity
    n0 : float
        n20, Density on axis
    T0 : float
        keV, Temperature on axis


    Outputs
    ------
    P : float
        MW, transparency factor

    Notes
    ------
    βT used here is not the β of a plasma (ratio of plasma pressure to
    magnetic pressure). Rather, it is a peaking parameter in the
    temperature profile as Albajar [1]_ writes:

    .. math::

       T(\rho) = (T_0 - T_a)(1 - \rho^{\beta_T})^{\alpha_T} + T_a.

    Note that here the edge electron temperature, Ta, is fixed to 1 keV.
    This profile is similar to the pedestal profile with the pedestal
    at the edge.

    References
    ----------
    .. [1] Albajar, F.; Johner, J.; Granata, G.
       Improved Calculation of Synchrotron Radiation Losses in Realistic
       Tokamak Plasmas. Nucl. Fusion 2001, 41 (6), 665–678.
       https://doi.org/10.1088/0029-5515/41/6/301.

    """

    def setup(self):
        self.add_input("A", desc="aspect ratio")
        self.add_input("a0", units="m", desc="minor radius")
        self.add_input("κ", desc="elongation")

        self.add_input("αn", desc="density peaking parameter")
        self.add_input("αT", desc="temperature peaking parameter")
        self.add_input("βT", desc="temperature peaking parameter for ρ")

        self.add_input("Bt", units="T", desc="toroidal magnetic field")
        self.add_input("pa", desc="optical thickness")
        self.add_input("r", desc="wall reflectivity")
        self.add_input("n0", units="n20", desc="center density")
        self.add_input("T0", units="keV", desc="center temperature")

        self.add_output("P", units="MW", desc="transparency factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        a0 = inputs["a0"]
        κ = inputs["κ"]

        αn = inputs["αn"]
        αT = inputs["αT"]
        βT = inputs["βT"]

        Bt = inputs["Bt"]
        pa = inputs["pa"]
        r = inputs["r"]
        n0 = inputs["n0"]
        T0 = inputs["T0"]

        term1 = (αn + 3.87 * αT + 1.46)**(-0.79) * (1.98 + αT)**1.36 * βT**2.14
        term2 = (βT**1.53 + 1.87 * αT - 0.16)**(-1.33)
        K = term1 * term2
        G = 0.93 * (1 + 0.85 * np.exp(-0.82 * A))
        factor1 = 3.84 * 10**(-8) * (1 - r)**(1 / 2) * A * a0**(2.38) * κ**0.79
        factor2 = Bt**2.62 * n0**0.38 * T0 * (16 + T0)**2.61
        factor3 = (1 + 0.12 * T0 / pa**0.41)**(-1.51)

        outputs["P"] = factor1 * factor2 * factor3 * K * G

    def setup_partials(self):
        self.declare_partials("P", ["A", "a0", "κ", "αn", "αT", "βT",
                                    "Bt", "pa", "r", "n0", "T0"])

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        a0 = inputs["a0"]
        κ = inputs["κ"]

        αn = inputs["αn"]
        αT = inputs["αT"]
        βT = inputs["βT"]

        Bt = inputs["Bt"]
        pa = inputs["pa"]
        r = inputs["r"]
        n0 = inputs["n0"]
        T0 = inputs["T0"]

        term1 = (αn + 3.87 * αT + 1.46)**(-0.79) * (1.98 + αT)**1.36 * βT**2.14
        term2 = (βT**1.53 + 1.87 * αT - 0.16)**(-1.33)
        K = term1 * term2
        G = 0.93 * (1 + 0.85 * np.exp(-0.82 * A))
        factor1 = 3.84 * 10**(-8) * (1 - r)**(1 / 2) * A * a0**(2.38) * κ**0.79
        factor2 = Bt**2.62 * n0**0.38 * T0 * (16 + T0)**2.61
        factor3 = (1 + 0.12 * T0 / pa**0.41)**(-1.51)

        a1 = 3.84 * 10**(-8) * (1 - r)**(1 / 2) * a0**(2.38) * κ**0.79
        g1 = 0.93 * 0.85 * -0.82 * np.exp(-0.82 * A)
        J["P", "A"] = (a1 * G + factor1 * g1) * factor2 * factor3 * K

        f1 = 3.84 * 10**(-8) * (1 - r)**(0.5) * A * 2.38 * a0**(1.38) * κ**0.79
        J["P", "a0"] = f1 * factor2 * factor3 * K * G

        f2 = 3.84 * 10**(-8) * (1 - r)**(0.5) * A * a0**(2.38) * 0.79
        J["P", "κ"] = f2 * κ**(-0.21) * factor2 * factor3 * K * G

        f3 = (1.98 + αT)**1.36 * βT**2.14 * term2 * G * factor1 * factor2
        J["P", "αn"] = -0.79 * (αn + 3.87 * αT + 1.46)**(-1.79) * f3 * factor3

        f4 = 3.87 * (1.98 + αT)**1.36 * βT**2.14 * term2
        n1 = -0.79 * (αn + 3.87 * αT + 1.46)**(-1.79) * f4
        n2 = (αn + 3.87 * αT +
              1.46)**(-0.79) * 1.36 * (1.98 + αT)**0.36 * βT**2.14 * term2
        n3 = term1 * -1.33 * (βT**1.53 + 1.87 * αT - 0.16)**(-2.33) * 1.87
        J["P", "αT"] = (n1 + n2 + n3) * G * factor1 * factor2 * factor3

        n4 = (αn + 3.87 * αT +
              1.46)**(-0.79) * (1.98 + αT)**1.36 * 2.14 * βT**1.14 * term2
        n5 = -1.33 * (βT**1.53 +
                      1.87 * αT - 0.16)**(-2.33) * 1.53 * βT**0.53 * term1
        J["P", "βT"] = (n4 + n5) * G * factor1 * factor2 * factor3

        f5 = 2.62 * Bt**1.62 * n0**0.38 * T0 * (16 + T0)**2.61
        J["P", "Bt"] = factor1 * f5 * factor3 * K * G

        n6 = -1.51 * (1 + 0.12 * T0 / pa**0.41)**(-2.51)
        n7 = 0.12 * T0 * -0.41 * pa**(-1.41)
        J["P", "pa"] = factor1 * factor2 * n6 * n7 * K * G

        f6 = 3.84 * 10**(-8) * -0.5 * (1 - r)**(-0.5) * A * a0**(2.38)
        J["P", "r"] = f6 * κ**0.79 * factor2 * factor3 * K * G

        f7 = Bt**2.62 * 0.38 * n0**(-0.62) * T0 * (16 + T0)**2.61
        J["P", "n0"] = factor1 * f7 * factor3 * K * G

        n8 = Bt**2.62 * n0**0.38 * (16 + T0)**1.61 * (T0 * 2.61 + 16 + T0)
        n9 = -1.51 * (1 + 0.12 * T0 / pa**0.41)**(-2.51) * 0.12 / pa**0.41
        J["P", "T0"] = factor1 * (factor2 * n9 + n8 * factor3) * K * G


class Synchrotron(om.Group):
    r"""Model for synchrotron radiation

    Notes
    -----
    Calculates synchrotron radiation from fit from Albajar [1]_, given by:

    .. math::
       P_{syn, r} = &3.84 \times 10^{-8}(1-r)^{1/2} Ra^{1.38}\kappa^{0.79} \\
           &\times B_t^{2.62}n_0^{0.38}T_0(16+T_0)^{2.61} \\
           &\times \left(1+0.12\frac{T_0}{p_a^{0.41}}\right)^{-1.51} \\
           &\times K(\alpha_n, \alpha_T, \beta_T) G(A)

    where K and G are profile and aspect ratio factors, respectively.

    The optical thickness of the plasma, as given in [1]_, is

    .. math::

       p_a = 6.04 \times 10^3 \frac{an_0}{B_t}.

    In this group, additional option for implementation of triangularity is
    implemented. This is based on the potentially plausible assumption that the
    relationship between synchrotron power and surface area is approximately
    linear. This should challenged and tested in further work.

    References
    ----------
    .. [1] Albajar, F.; Johner, J.; Granata, G.
       Improved Calculation of Synchrotron Radiation Losses in Realistic
       Tokamak Plasmas. Nucl. Fusion 2001, 41 (6), 665–678.
       https://doi.org/10.1088/0029-5515/41/6/301.

    """

    def initialize(self):
        self.options.declare("implement_triangularity", default=True)

    def setup(self):
        triangularity = self.options["implement_triangularity"]

        self.m = 6.04e3

        optical_string = f"pa = {self.m} * a0 * n0 / Bt"
        self.add_subsystem("optical_thickness",
                           om.ExecComp([optical_string],
                                       a0={"units": "m"},
                                       n0={"units": "n20"},
                                       Bt={"units": "T"}),
                           promotes_inputs=["a0", "n0", "Bt"],
                           promotes_outputs=["pa"])

        if triangularity:
            self.add_subsystem("synchrotron_fit",
                               SynchrotronFit(),
                               promotes_inputs=["A", "a0", "κ", "αn",
                                                "αT", "βT", "Bt", "pa",
                                                "r", "n0", "T0"],
                               promotes_outputs=[("P", "P0")])

            θ = np.linspace(0, 2 * pi, 1, endpoint=False)
            self.add_subsystem("ivc",
                               om.IndepVarComp("θ", val=θ),
                               promotes_outputs=["*"])
            self.add_subsystem("major_radius",
                               om.ExecComp("R0=A * a0",
                                           R0={"units": "m"},
                                           A={"units": None},
                                           a0={"units": "m"}),
                               promotes_inputs=["A", "a0"],
                               promotes_outputs=["R0"])
            self.add_subsystem("elliptical_surface_area",
                               SauterGeometry(),
                               promotes_inputs=["R0", "A", ("a", "a0"),
                                                "κ", "θ"],
                               promotes_outputs=[("S", "S0")])

            self.add_subsystem("surface_area",
                               SauterGeometry(),
                               promotes_inputs=["R0", "A", ("a", "a0"), "κ",
                                                "δ", "θ"],
                               promotes_outputs=["S"])

            self.add_subsystem("synchrotron_power",
                               om.ExecComp("P= P0 * S / S0",
                                           P0={"units": "MW"},
                                           P={"units": "MW"},
                                           S={"units": "m**2"},
                                           S0={"units": "m**2"}),
                               promotes_inputs=["P0", "S", "S0"],
                               promotes_outputs=["P"])

        else:
            self.add_subsystem("synchrotron_fit",
                               SynchrotronFit(),
                               promotes_inputs=["A", "a0", "κ", "αn",
                                                "αT", "βT", "Bt", "pa",
                                                "r", "n0", "T0"],
                               promotes_outputs=["P"])
            self.add_subsystem("ignore",
                               om.ExecComp("ignore=0 * delta"),
                               promotes_inputs=[("delta", "δ")])


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = Synchrotron(implement_triangularity=False)

    prob.setup()
    prob.set_val("A", 3)
    prob.set_val("a0", 2.7, units='m')
    prob.set_val("κ", 1.9)
    prob.set_val("δ", 0.0)

    prob.set_val("αn", 2)
    prob.set_val("αT", 3)
    prob.set_val("βT", 2)

    prob.set_val("Bt", 6.8, units='T')
    prob.set_val("r", 0.0)
    prob.set_val("n0", 1.36, units='n20')
    prob.set_val("T0", 45, units='keV')

    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True, units=True)
    all_outputs = prob.model.list_outputs(values=True, units=True)
