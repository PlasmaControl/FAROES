from faroes.configurator import UserConfigurator
import openmdao.api as om
import faroes.util as util
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy.sparse


class SauterGeometry(om.ExplicitComponent):
    r"""Describes a D-shaped plasma based on Sauter's formulas.

    .. math::

        R &= R_0 + a\cos(\theta + \delta\sin(\theta) - \xi\sin(2\theta)) \\
        Z &= \kappa * a\sin(\theta + \xi\sin(2\theta))

    Other properties are determined using standard geometry formulas:

    .. math::

        R_0 &= \frac{R_\mathrm{max} + R_\mathrm{min}}{2} \\
        a &= \frac{R_\mathrm{max} - R_\mathrm{min}}{2} \\
        \epsilon &= a / R_0 \\
        \kappa &= \frac{Z_\mathrm{max} - Z_\mathrm{min}}{R_\mathrm{max} - R_\mathrm{min}} \\

    Inputs
    ------
    R0 : float
        m, major radius
    A : float
        Aspect ratio
    a : float
        m, minor radius
    κ : float
        elongation
    δ : float
        triangularity
    ξ : float
        related to the plasma squareness
    θ : array
        poloidal angles at which to evaluate the plasma boundary

    Outputs
    -------
    R : array
        m, radial locations of plasma boundary
    Z : array
        m, vertical locations of plasma boundary
    Z0 : float
        m, vertical location of magnetic axis (always 0)
    b : float
        m, minor radius in vertical direction
    ε : float
        inverse aspect ratio
    κa : float
        effective elongation, S_c / (π a^2),
    full plasma height : float
        m, twice b
    S_c : float
        m**2, poloidal cross-section area
    S : float
        m**2, surface area
    V : float
        m**3, volume of the elliptical torus
    R_min : float
        m, innermost plasma radius at midplane
    R_max : float
        m, outermost plasma radius at midplane
    L_pol : float
        m, Poloidal circumference
    w07 : float
        Sauter 70% width parameter



    Reference
    ---------
    Fusion Engineering and Design 112, 633-645 (2016);
    http://dx.doi.org/10.1016/j.fusengdes.2016.04.033

    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        if self.options["config"] is not None:
            self.config = self.options["config"]

        self.add_input("R0", units="m", desc="Major radius")
        self.add_input("A", desc="Aspect ratio")
        self.add_input("κ", desc="Elongation")
        self.add_input("δ", desc="Triangularity", val=0)
        self.add_input("ξ", desc="Related to the plasma squareness", val=0)
        self.add_input("θ",
                       shape_by_conn=True,
                       desc="Poloidal locations to evaluate (R, Z)")

        self.add_output("R",
                        units="m", lower=0,
                        copy_shape="θ",
                        desc="Radial locations of plasma boundary")
        self.add_output("Z",
                        units="m",
                        copy_shape="θ",
                        desc="Vertical locations of plasma boundary")
        self.add_output("Z0",
                        units="m",
                        val=0,
                        desc="vertical location of magnetic axis")
        self.add_output("a", units="m", desc="Minor radius")
        self.add_output("b", units="m", desc="Minor radius height")
        self.add_output("ε", desc="Inverse aspect ratio")
        # self.add_output("κa", desc="Effective elongation")
        self.add_output("full_plasma_height",
                        units="m", ref=10,
                        desc="Full plasma height")
        self.add_output("S_c",
                        units="m**2", ref=10,
                        desc="Poloidal cross-section area")
        self.add_output("S", units="m**2", ref=100, desc="Surface area")
        self.add_output("V", units="m**3", ref=100, desc="Volume")
        self.add_output("L_pol",
                        units="m",
                        desc="Poloidal circumference",
                        lower=0,
                        ref=10)
        self.add_output("R_min",
                        units="m", lower=0,
                        desc="Inner radius of plasma at midplane")
        self.add_output("R_max",
                        units="m", lower=0,
                        desc="outer radius of plasma at midplane")
        self.add_output("w07", desc="Sauter 70% width parameter")
        self.add_output("κa", lower=0, desc="effective elongation")

    def compute(self, inputs, outputs):
        outputs["Z0"] = 0

        R0 = inputs["R0"]
        A = inputs["A"]
        κ = inputs["κ"]
        δ = inputs["δ"]
        ξ = inputs["ξ"]

        θ = inputs["θ"]
        a = R0 / A
        outputs["a"] = a

        # Sauter equation (C.2)
        # but using the 'alternate expression for a quadratic root'
        # 2c / (-b + √(b² - 4 a c))
        θ07 = np.arcsin(0.7) - (2 * ξ) / (1 + (1 + 8 * ξ**2)**(1 / 2))

        # Sauter equation (C.5)
        wanal_07 = np.cos(θ07 - ξ * np.sin(2 * θ07)/ (0.51)**(1/2))  \
                * (1 - 0.49 * δ**2 / 2)
        w07 = wanal_07

        L_p = (2 * pi * a * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
               (1 + 0.2 * (w07 - 1)))

        outputs["L_pol"] = L_p

        # Equation (1)
        R = R0 + a * np.cos(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ))
        # Equation (2)
        Z = κ * a * np.sin(θ + ξ * np.sin(2 * θ))

        b = κ * a
        outputs["b"] = b
        outputs["full_plasma_height"] = 2 * b

        outputs["R"] = R
        outputs["Z"] = Z

        ε = 1 / A
        outputs["ε"] = ε

        # Equation (36)
        Ap = (2 * pi * R0) * (1 - 0.32 * δ * ε) * L_p

        # Equation (36); here Sφ is called S_c
        S_c = (pi * a**2 * κ) * (1 + 0.52 * (w07 - 1))
        outputs["κa"] = S_c / (pi * a**2)

        # Equation (36)
        V = (2 * pi * R0) * (1 - 0.25 * δ * ε) * S_c
        outputs["S_c"] = S_c
        outputs["S"] = Ap
        outputs["V"] = V

        outputs["R_min"] = R0 - a
        outputs["R_max"] = R0 + a
        outputs["w07"] = w07

    def setup_partials(self):
        size = self._get_var_meta("θ", "size")
        self.declare_partials("ε", ["A"])
        self.declare_partials("a", ["R0", "A"])
        self.declare_partials("full_plasma_height", ["R0", "A", "κ"])
        self.declare_partials("R_min", ["R0", "A"])
        self.declare_partials("R_max", ["R0", "A"])

        self.declare_partials("b", ["R0", "A", "κ"])
        self.declare_partials("w07", ["δ", "ξ"])

        self.declare_partials("L_pol", ["R0", "A", "κ", "δ", "ξ"])
        self.declare_partials("S", ["R0", "A", "κ", "δ", "ξ"])
        self.declare_partials("S_c", ["R0", "A", "κ", "δ", "ξ"])
        self.declare_partials("V", ["R0", "A", "κ", "δ", "ξ"])

        self.declare_partials("κa", ["R0", "A", "κ", "δ", "ξ"])

        self.declare_partials("R", ["R0", "A", "δ", "ξ"])
        self.declare_partials("R", ["θ"],
                              val=scipy.sparse.eye(size, format="csc"))

        self.declare_partials("Z", ["R0", "A", "κ", "ξ"], method="cs")
        self.declare_partials("Z", ["θ"],
                              val=scipy.sparse.eye(size, format="csc"))

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        R0 = inputs["R0"]
        ξ = inputs["ξ"]

        R0 = inputs["R0"]
        A = inputs["A"]
        κ = inputs["κ"]
        δ = inputs["δ"]
        ξ = inputs["ξ"]

        θ = inputs["θ"]

        a = R0 / A
        b = κ * a
        da_dA = -R0 / A**2
        da_dR0 = 1 / A
        ε = 1 / A
        J["ε", "A"] = -1 / A**2
        J["a", "R0"] = da_dR0
        J["a", "A"] = da_dA
        J["b", "κ"] = a
        J["b", "R0"] = κ * J["a", "R0"]
        J["b", "A"] = κ * J["a", "A"]
        J["full_plasma_height", "κ"] = 2 * J["b", "κ"]
        J["full_plasma_height", "R0"] = 2 * J["b", "R0"]
        J["full_plasma_height", "A"] = 2 * J["b", "A"]

        J["R_min", "R0"] = 1 - 1 / A
        J["R_min", "A"] = R0 / A**2
        J["R_max", "R0"] = 1 + 1 / A
        J["R_max", "A"] = -R0 / A**2

        # Sauter equation (C.2)
        # but using the 'alternate expression for a quadratic root'
        # 2c / (-b + √(b² - 4 a c))
        θ07 = np.arcsin(0.7) - (2 * ξ) / (1 + (1 + 8 * ξ**2)**(1 / 2))
        # Sauter equation (C.5)
        wanal_07 = np.cos(θ07 - ξ * np.sin(2 * θ07)/ (0.51)**(1/2))  \
                * (1 - 0.49 * δ**2 / 2)
        w07 = wanal_07

        dθ_dξ = -2 / (1 + 8 * ξ**2 + (1 + 8 * ξ**2)**(1 / 2))

        dw07_dξ = (np.sin(2 * θ07) * np.sin(θ07 - ξ * np.sin(2 * θ07) /
                                            (0.51)**(1 / 2))
                   ) * (1 - 0.49 * δ**2 / 2) / 0.51**(1 / 2)
        dw07_dθ07 = -((1 - 2 * ξ * np.cos(2 * θ07) / 0.51**
                       (1 / 2)) * np.sin(θ07 - ξ * np.sin(2 * θ07) / 0.51**
                                         (1 / 2))) * (1 - 0.49 * δ**2 / 2)

        dw07_dδ = -0.49 * δ * np.cos(θ07 - ξ * np.sin(2 * θ07) / 0.51**(1 / 2))
        J["w07", "δ"] = dw07_dδ
        J["w07", "ξ"] = dw07_dξ + dw07_dθ07 * dθ_dξ

        L_p = (2 * pi * a * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
               (1 + 0.2 * (w07 - 1)))
        # partials
        dLp_dR0 = (2 * pi * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
                   (1 + 0.2 * (w07 - 1))) / A
        dLp_dA = -(2 * pi * R0 * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
                   (1 + 0.2 * (w07 - 1))) / A**2
        dLp_dκ = 11 / 20 * (2 * pi * a * (1 + 0.08 * δ**2) * (1 + 0.2 *
                                                              (w07 - 1)))
        dLp_dδ = 0.16 * δ * (2 * pi * a * (1 + 0.55 * (κ - 1)) * (1 + 0.2 *
                                                                  (w07 - 1)))
        dLp_dw07 = 0.2 * (2 * pi * a * (1 + 0.08 * δ**2) * (1 + 0.55 *
                                                            (κ - 1)))

        J["L_pol", "A"] = dLp_dA
        J["L_pol", "R0"] = dLp_dR0
        J["L_pol", "κ"] = dLp_dκ
        J["L_pol", "δ"] = dLp_dδ + dLp_dw07 * J["w07", "δ"]
        J["L_pol", "ξ"] = dLp_dw07 * J["w07", "ξ"]

        # partials
        dAp_dR0 = 2 * pi * (1 - 0.32 * δ * ε) * L_p
        dAp_dLp = 2 * pi * R0 * (1 - 0.32 * δ * ε)
        dAp_dA = 2 * pi * R0 * (+0.32 * δ * 1 / A**2) * L_p
        dAp_dδ = 2 * pi * R0 * (-0.32 * ε) * L_p

        J["S", "A"] = dAp_dA + dAp_dLp * J["L_pol", "A"]
        J["S", "R0"] = dAp_dR0 + dAp_dLp * J["L_pol", "R0"]
        J["S", "κ"] = dAp_dLp * J["L_pol", "κ"]
        J["S", "δ"] = dAp_dδ + dAp_dLp * J["L_pol", "δ"]
        J["S", "ξ"] = dAp_dLp * J["L_pol", "ξ"]

        S_c = (pi * a**2 * κ) * (1 + 0.52 * (w07 - 1))
        dSc_dw07 = (pi * a**2 * κ) * 0.52

        dSc_da = 2 * pi * a * κ * (1 + 0.52 * (w07 - 1))
        J["S_c", "A"] = dSc_da * da_dA
        J["S_c", "R0"] = dSc_da * da_dR0
        J["S_c", "κ"] = (pi * a**2) * (1 + 0.52 * (w07 - 1))
        J["S_c", "δ"] = dSc_dw07 * J["w07", "δ"]
        J["S_c", "ξ"] = dSc_dw07 * J["w07", "ξ"]

        dκa_da = -2 * S_c / (pi * a**3)
        dκa_dSc = 1 / (pi * a**2)
        J["κa", "A"] = dκa_da * da_dA + dκa_dSc * J["S_c", "A"]
        J["κa", "R0"] = dκa_da * da_dR0 + dκa_dSc * J["S_c", "R0"]
        J["κa", "κ"] = dκa_dSc * J["S_c", "κ"]
        J["κa", "δ"] = dκa_dSc * J["S_c", "δ"]
        J["κa", "ξ"] = dκa_dSc * J["S_c", "ξ"]

        V = (2 * pi * R0) * (1 - 0.25 * δ * 1 / A) * S_c
        dV_dR0 = (2 * pi) * (1 - 0.25 * δ * 1 / A) * S_c
        dV_dA = (2 * pi * R0) * (-0.25 * δ * -1 / A**2) * S_c
        dV_dδ = (2 * pi * R0) * (-0.25 * 1 / A) * S_c
        dV_dSc = (2 * pi * R0) * (1 - 0.25 * δ * 1 / A)
        J["V", "R0"] = dV_dR0 + dV_dSc * J["S_c", "R0"]
        J["V", "A"] = dV_dA + dV_dSc * J["S_c", "A"]
        J["V", "κ"] = dV_dSc * J["S_c", "κ"]
        J["V", "δ"] = dV_dδ + dV_dSc * J["S_c", "δ"]
        J["V", "ξ"] = dV_dSc * J["S_c", "ξ"]

        J["R",
          "A"] = -(R0 * np.cos(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ)) / A**2)
        J["R", "R0"] = 1 + np.cos(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ)) / A
        J["R", "δ"] = -(R0 * np.sin(θ) *
                        np.sin(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ))) / A
        J["R", "ξ"] = R0 * np.sin(
            2 * θ) * np.sin(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ)) / A

        dR_dθ = -a * (1 + δ * np.cos(θ) - 2 * ξ * np.cos(2 * θ)
                      ) * np.sin(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ))

        size = self._get_var_meta("θ", "size")
        dm = scipy.sparse.dia_matrix((dR_dθ, [0]), shape=(size, size))
        J["R", "θ"] = dm

        J["Z", "A"] = -R0 * κ * np.sin(θ + ξ * np.sin(2 * θ)) / A**2
        J["Z", "R0"] = κ * np.sin(θ + ξ * np.sin(2 * θ)) / A
        J["Z", "κ"] = a * np.sin(θ + ξ * np.sin(2 * θ))
        J["Z", "ξ"] = a * κ * np.cos(θ + ξ * np.sin(2 * θ)) * np.sin(2 * θ)
        dZ_dθ = a * κ * (1 + 2 * ξ * np.cos(2 * θ)) * np.cos(θ +
                                                             ξ * np.sin(2 * θ))
        dm = scipy.sparse.dia_matrix((dZ_dθ, [0]), shape=(size, size))
        J["Z", "θ"] = dm

    def plot(self, ax=plt.subplot(111), **kwargs):
        label = 'R0={}, a={}, κ={}, δ={}, ξ={}'.format(
            self.get_val('R0')[0],
            self.get_val('a')[0],
            self.get_val('κ')[0],
            self.get_val('δ')[0],
            self.get_val('ξ')[0])

        ax.plot(self.get_val('R0'), self.get_val('Z0'), marker='x', **kwargs)
        ax.plot(self.get_val('R'), self.get_val('Z'), label=label, **kwargs)

        ax.axis('equal')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(label)


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()

    θ = np.linspace(0, 2 * pi, 180, endpoint=False)

    sg = SauterGeometry(config=uc)

    prob.model.add_subsystem("ivc",
                             om.IndepVarComp("θ", val=θ),
                             promotes_outputs=["*"])
    prob.model.add_subsystem("geom", sg, promotes_inputs=["*"])

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('A', 1.6)
    prob.set_val('κ', 2.7)
    prob.set_val('δ', 0.5)
    prob.set_val('ξ', 0.3)

    prob.run_driver()
    prob.model.list_inputs()
    prob.model.list_outputs()

    # test by plotting
    # sg.plot(c='k', ls='-')
    # plt.show()
