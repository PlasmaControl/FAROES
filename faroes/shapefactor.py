from faroes.sauter_plasma import SauterGeometry

import openmdao.api as om
from scipy.constants import pi
from scipy.special import hyp0f1, gamma, jv, digamma
import numpy as np


class ConstProfile(om.ExplicitComponent):
    r"""Model for constant temperature and density profiles

    Computes the shapefactor, S, for constant temperature and
    density profiles, given by

    .. math::
       S = \frac{\int dV}{a_0^3},

    where a0 is the minor radius, V is the volume of the plasma
    distribution.

    Here, constant (temperature and density) profiles imply

    .. math::
       f(\rho) = f_0.

    Inputs
    ------
    A : float
        None, Aspect ratio (R0 / a0)
    δ0 : float
        None, Triangularity of last closed flux surface (LCFS)
    κ : float
        None, Elongation of plasma distribution shape


    Outputs
    ------
    S : float
        None, shape factor
    """
    def setup(self):
        self.add_input("A", desc="aspect ratio")
        self.add_input("δ0", val=0., desc="LCFS triangularity")
        self.add_input("κ", desc="elongation")

        self.add_output("S", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]

        term1 = 4 * A * (jv(0, δ0) + jv(2, δ0))
        term2 = (jv(1, 2 * δ0) + jv(3, 2 * δ0))
        outputs["S"] = pi**2 * κ / 2 * (term1 - term2)

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ"])

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]

        J["S", "A"] = 2 * pi**2 * κ * (jv(0, δ0) + jv(2, δ0))

        t1 = 2 * A * (jv(1, δ0) + jv(3, δ0))
        t2 = (jv(0, 2 * δ0) - jv(4, 2 * δ0))
        J["S", "δ0"] = -pi**2 * κ / 2 * (t1 + t2)

        t3 = 4 * A * (jv(0, δ0) + jv(2, δ0))
        t4 = (jv(1, 2 * δ0) + jv(3, 2 * δ0))
        J["S", "κ"] = pi**2 / 2 * (t3 - t4)


class ParabProfileConstTriang(om.ExplicitComponent):
    r"""Model for parabolic profiles and constant triangularity

    Computes the shapefactor, S, for parabolic temperature and
    density profiles, given by

    .. math::
       S = \frac{\int (1-\rho^2)^\alpha dV}{a_0^3},

    where a0 is the minor radius, V is the volume of the plasma
    distribution.

    Here, parabolic (temperature and density) profiles imply

    .. math::
       f(\rho) = f_0 (1-\rho^2)^\alpha.

    and constant triangularity means

    .. math::
       \delta(\rho) = \delta_0.

    Inputs
    ------
    A : float
        None, Aspect ratio (R0 / a0)
    δ0 : float
        None, Triangularity of border curve of plasma distribution
    κ : float
        None, Elongation of plasma distribution shape
    α : float
        None, exponent in density and temperature profiles


    Outputs
    ------
    S : float
        None, shape factor
    """
    def setup(self):
        self.add_input("A", desc="major radius")
        self.add_input("δ0", val=0., desc="border triangularity")
        self.add_input("κ", desc="elongation")
        self.add_input("α", desc="exponent for profiles")

        self.add_output("S", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        frac1 = 2 * pi**2 * A * κ / (1 + α) * (jv(0, δ0) + jv(2, δ0))
        frac2_1 = 3 * pi**(5 / 2) * κ / 8 * (jv(1, 2 * δ0) + jv(3, 2 * δ0))
        frac2_2 = gamma(1 + α) / (gamma(5 / 2 + α))
        outputs["S"] = frac1 - frac2_1 * frac2_2

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ", "α"])

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        J["S", "A"] = 2 * pi**2 * κ / (1 + α) * (jv(0, δ0) + jv(2, δ0))

        frac1 = -8 * A * (jv(1, δ0) + jv(3, δ0)) / (1 + α)
        frac2_1 = 3 * pi**(1 / 2) * (jv(0, 2 * δ0) - jv(4, 2 * δ0))
        frac2_2 = gamma(1 + α) / (gamma(5 / 2 + α))
        J["S", "δ0"] = pi**2 * κ / 8 * (frac1 - frac2_1 * frac2_2)

        frac3 = 2 * pi**2 * A / (1 + α) * (jv(0, δ0) + jv(2, δ0))
        frac4_1 = 3 * pi**(5 / 2) / 8 * (jv(1, 2 * δ0) + jv(3, 2 * δ0))
        frac4_2 = gamma(1 + α) / (gamma(5 / 2 + α))
        J["S", "κ"] = frac3 - frac4_1 * frac4_2

        frac5_1 = -16 * A * jv(0, δ0) / (1 + α)**2
        frac5_2 = 16 * A * jv(2, δ0) / (1 + α)**2
        num3_1 = 3 * pi**(1 / 2) * (jv(1, 2 * δ0) + jv(3, 2 * δ0))
        num3_2 = gamma(1 + α) * (digamma(1 + α) - digamma(5 / 2 + α))
        J["S", "α"] = pi**2 * κ / 8 * ((frac5_1 - frac5_2) - num3_1 * num3_2 /
                                       (gamma(5 / 2 + α)))


class ParabProfileLinearTriang(om.ExplicitComponent):
    r"""Model for parabolic profiles and constant triangularity

    Computes the shapefactor, S, for constant temperature and
    density profiles, given by

    .. math::
       S = \frac{\int (1-\rho^2)^\alpha dV}{a_0^3},

    where a0 is the minor radius, V is the volume of the plasma
    distribution.

    Here, parabolic (temperature and density) profiles imply

    .. math::
       f(\rho) = f_0 (1-\rho^2)^\alpha.

    and linear triangularity means

    .. math::
       \delta(\rho) = \delta_0 \rho.

    Inputs
    ------
    A : float
        None, Aspect ratio (R0 / a0)
    δ0 : float
        None, Triangularity of border curve of plasma distribution
    κ : float
        None, Elongation of plasma distribution shape
    α : float
        None, exponent in density and temperature profiles


    Outputs
    ------
    S : float
        None, shape factor
    """
    def setup(self):
        self.add_input("A", desc="major radius")
        self.add_input("δ0", val=0., desc="border triangularity")
        self.add_input("κ", desc="elongation")
        self.add_input("α", desc="exponent for profiles")

        self.add_output("S", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        frac1 = 2 * A * hyp0f1(2 + α, -δ0**2 / 4) / gamma(2 + α)
        frac2 = δ0 * hyp0f1(3 + α, -δ0**2) / gamma(3 + α)
        outputs["S"] = pi**2 * κ * gamma(1 + α) * (frac1 - frac2)

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ"])
        self.declare_partials("S", ["α"], method="fd", form="central")

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        num1 = pi**2 * κ * gamma(1 + α) * 2
        J["S", "A"] = num1 * hyp0f1(2 + α, -δ0**2 / 4) / gamma(2 + α)

        term1 = hyp0f1(3 + α, -δ0**2) / gamma(3 + α)
        term2 = δ0 * (A * hyp0f1(3 + α, -δ0**2 / 4) / gamma(3 + α) -
                      2 * δ0 * hyp0f1(4 + α, -δ0**2) / gamma(4 + α))
        J["S", "δ0"] = -pi**2 * κ * gamma(1 + α) * (term1 + term2)

        frac3 = 2 * A * hyp0f1(2 + α, -δ0**2 / 4) / gamma(2 + α)
        frac4 = δ0 * hyp0f1(3 + α, -δ0**2) / gamma(3 + α)
        J["S", "κ"] = pi**2 * gamma(1 + α) * (frac3 - frac4)


class PeakingFactor(om.Group):
    r"""Computes Peaking factor
    For a particular function, F, of ρ, the peaking factor is defined as
    the ratio of the value at the center to the volume-average. It is given
    by

    .. math::
       \frac{F_0}{\langle F(\rho) \rangle} = \frac{V}{S_F * a_0**3}

    for constant and parabolic profiles, where V is the volume, a0 is the minor
    radius and S is the corresponding shape factor.

    Inputs
    ------
    A : float
        Aspect ratio (R0 / a0)
    a0 : float
        m, minor radius
    δ0 : float
        Triangularity of plasma distribution
    κ : float
        Elongation of plasma distribution
    α : float
        None, exponent in profile

    Outputs
    -------
    pf : float
        None, peaking factor

    Notes
    -----
    This current implementation is only for the case of parabolic profile
    and constant triangularity.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
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

        self.add_subsystem("volume",
                           SauterGeometry(),
                           promotes_inputs=["R0", "A", ("a", "a0"), "κ",
                                            ("δ", "δ0"), "θ"],
                           promotes_outputs=["V"])

        self.add_subsystem("shape_factor",
                           ParabProfileConstTriang(),
                           promotes_inputs=["A", "δ0", "κ", "α"],
                           promotes_outputs=["S"])

        self.add_subsystem("peaking_factor",
                           om.ExecComp("pf= V / (S * a0**3)",
                                       V={"units": "m**3"},
                                       a0={"units": "m"}),
                           promotes_inputs=["V", "S", "a0"],
                           promotes_outputs=["pf"])


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = PeakingFactor()

    prob.setup(force_alloc_complex=True)
    prob.set_val("A", 5 / 2)
    prob.set_val("a0", 2)
    prob.set_val("δ0", 0.2)
    prob.set_val("κ", 1)
    prob.set_val("α", 2)
    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True, units=True)
    all_outputs = prob.model.list_outputs(values=True, units=True)
