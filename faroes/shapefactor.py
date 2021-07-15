import openmdao.api as om

from scipy.constants import pi
from scipy.special import hyp0f1, gamma, jv, digamma
from scipy.integrate import quad


class ConstProfile(om.ExplicitComponent):
    r"""Model for constant temperature and density profiles

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
        self.add_input("δ0", desc="LCFS triangularity")
        self.add_input("κ", desc="elongation")

        self.add_output("S", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]

        outputs["S"] = pi**2 * κ / 2 * (4 * A * (jv(0, δ0) + jv(2, δ0)) - (jv(1, 2 * δ0) + jv(3, 2 * δ0)))

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ"])

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]

        J["S", "A"] = 2 * pi**2 * κ * (jv(0, δ0) + jv(2, δ0))
        J["S", "δ0"] = -pi**2 * κ / 2 * (2 * A * (jv(1, δ0) + jv(3, δ0)) + (jv(0, 2 * δ0) - jv(4, 2 * δ0)))
        J["S", "κ"] = pi**2 / 2 * (4 * A * (jv(0, δ0) + jv(2, δ0)) - (jv(1, 2 * δ0) + jv(3, 2 * δ0)))


class ParabProfileConstTriang(om.ExplicitComponent):
    r"""Model for parabolic profiles and constant triangularity

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
        self.add_input("δ0", desc="border triangularity")
        self.add_input("κ", desc="elongation")
        self.add_input("α", desc="exponent for profiles")

        self.add_output("S", desc="shape factor")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        frac1 = 2 * pi**2 * A * κ / (1 + α) * (jv(0, δ0) + jv(2, δ0))
        frac2 = 3 * pi**(5 / 2) * κ / 8 * (jv(1, 2 * δ0) + jv(3, 2 * δ0)) * gamma(1 + α) / (gamma(5 / 2 + α))
        outputs["S"] = frac1 - frac2

    def setup_partials(self):
        self.declare_partials("S", ["A", "δ0", "κ", "α"])

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        J["S", "A"] = 2 * pi**2 * κ / (1 + α) * (jv(0, δ0) + jv(2, δ0))

        frac1 = -8 * A * (jv(1, δ0) + jv(3, δ0)) / (1 + α)
        frac2 = 3 * pi**(1 / 2) * (jv(0, 2 * δ0) - jv(4, 2 * δ0)) * gamma(1 + α)/(gamma(5 / 2 + α))
        J["S", "δ0"] = pi**2 * κ / 8 * (frac1 - frac2)

        frac3 = 2 * pi**2 * A / (1 + α) * (jv(0, δ0) + jv(2, δ0))
        frac4 = 3 * pi**(5 / 2) / 8 * (jv(1, 2 * δ0) + jv(3, 2 * δ0)) * gamma(1 + α) / (gamma(5 / 2 + α))
        J["S", "κ"] = frac3 - frac4

        frac5 = -16 * A * jv(0, δ0) / (1 + α)**2 - 16 * A * jv(2, δ0) / (1 + α)**2
        num3 = 3 * pi**(1 / 2) * (jv(1, 2 * δ0) + jv(3, 2 * δ0)) * gamma(1 + α) * (digamma(1 + α) - digamma(5 / 2 + α))
        J["S", "α"] = pi**2 * κ / 8 * (frac5 - num3 / (gamma(5 / 2 + α)))


class ParabProfileLinearTriang(om.ExplicitComponent):
    r"""Model for parabolic profiles and constant triangularity

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
        self.add_input("δ0", desc="border triangularity")
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
        self.declare_partials("S", ["α"], method="fd")

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        δ0 = inputs["δ0"]
        κ = inputs["κ"]
        α = inputs["α"]

        J["S", "A"] = pi**2 * κ * gamma(1 + α) * 2 * hyp0f1(2 + α, -δ0**2 / 4) / gamma(2 + α)

        term1 = hyp0f1(3 + α, -δ0**2) / gamma(3 + α)
        term2 = δ0 * (A * hyp0f1(3 + α, -δ0**2 / 4) / gamma(3 + α) - 2 * δ0 * hyp0f1(4 + α, -δ0**2) / gamma(4 + α))
        J["S", "δ0"] = -pi**2 * κ * gamma(1 + α) * (term1 + term2)

        frac3 = 2 * A * hyp0f1(2 + α, -δ0**2 / 4) / gamma(2 + α)
        frac4 = δ0 * hyp0f1(3 + α, -δ0**2) / gamma(3 + α)
        J["S", "κ"] = pi**2 * gamma(1 + α) * (frac3 - frac4)
