from faroes.configurator import UserConfigurator
from faroes.elliptical_plasma import MenardKappaScaling

import openmdao.api as om

from scipy.constants import pi

import numpy as np
from numpy import sin, cos


class SauterGeometry(om.ExplicitComponent):
    r"""Plasma shape from Sauter's formulas.

    Sauter's paper [1]_ describes plasma with a LCFS
    (last closed flux surface) shape parameterized by the angle around the
    magnetic axis :math:`\theta`,

    .. math::

        R &= R_0 + a\cos(\theta + \delta\sin(\theta) - \xi\sin(2\theta)) \\
        Z &= \kappa a\sin(\theta + \xi\sin(2\theta))

    where :math:`a` is the minor radius, :math:`\kappa` is elongation,
    :math:`\delta` is triangularity, and :math:`\xi` is "squareness".
    The basic shape properties are defined in the usual way,

    .. math::

        R_0 &= \frac{R_\mathrm{out} + R_\mathrm{in}}{2} \\
        a &= \frac{R_\mathrm{out} - R_\mathrm{in}}{2} \\
        \epsilon &= a / R_0 \\
        \kappa &= \frac{Z_\mathrm{top} - Z_\mathrm{bot}}{R_\mathrm{out} -
        R_\mathrm{in}}. \\

    Because the exact cross-sectional area :math:`S_c`, poloidal circumference
    :math:`L_\mathrm{pol}`, surface area :math:`A_p` and volume :math:`V`
    cannot be described by algebraic functions of the input parameters,
    Sauter provides best-fit formulas sampled from a wide range
    in parameter space.

    .. math::

       S_c &= (\pi a^2 \kappa)(1 + 0.52 (w_{07} - 1)) \\
        L_\mathrm{pol} &= (2 \pi a (1 + 0.55 (\kappa - 1))
           (1 + 0.08 δ^2)(1 + 0.2(w_{07} - 1))) \\
       A_p &= (2 \pi R_0)(1 - 0.32 \delta \epsilon) L_\mathrm{pol} \\
       V &= (2 \pi R_0)(1 - 0.25\delta\epsilon) S_c

    where :math:`w_{07}` is the plasma width at 0.7 of the
    maximum height. It is defined by

    .. math::

       \theta_{07} &= \sin^{-1}(0.7) - (2 \xi) / (1 + (1 + 8 \xi^2)^{1/2}) \\
       w_{07} &= \cos(\theta_{07}- \xi \sin(2 \theta_{07}) / 0.51^{1/2})
            (1 - 0.49 \delta^2/2).

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
        m, Radial locations on plasma boundary
    Z : array
        m, Vertical locations on plasma boundary
    Z0 : float
        m, Vertical location of magnetic axis (always 0)
    b : float
        m, Minor radius in vertical direction
    ε : float
        Inverse aspect ratio
    w07 : float
        Sauter 70% width parameter
    full plasma height : float
        m, Twice b
    S_c : float
        m**2, Poloidal cross-section area
    κa : float
        Effective elongation, S_c / (π a^2),
    S : float
        m**2, Swept-LCFS surface area
    V : float
        m**3, Plasma volume
    R_in : float
        m, innermost plasma radius at midplane
    R_out : float
        m, outermost plasma radius at midplane
    L_pol : float
        m, Poloidal circumference

    dR_dθ : array
        m, Derivative of R points w.r.t. θ
    dZ_dθ : array
        m, Derivative of Z points w.r.t. θ

    References
    ----------
    .. [1] Fusion Engineering and Design 112, 633-645 (2016);
       http://dx.doi.org/10.1016/j.fusengdes.2016.04.033
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        if self.options["config"] is not None:
            self.config = self.options["config"]

        self.add_input("R0", units="m", desc="Major radius")
        self.add_input("a", units="m", desc="Minor radius")
        self.add_input("A", desc="Aspect ratio")
        self.add_input("κ", desc="Elongation")
        self.add_input("δ", desc="Triangularity", val=0)
        self.add_input("ξ", desc="Related to the plasma squareness", val=0)
        self.add_input("θ",
                       shape_by_conn=True,
                       desc="Poloidal locations to evaluate (R, Z)")

        self.add_output("R",
                        units="m",
                        lower=0,
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
        self.add_output("b", units="m", desc="Minor radius height")
        self.add_output("ε", desc="Inverse aspect ratio")
        # self.add_output("κa", desc="Effective elongation")
        self.add_output("full_plasma_height",
                        units="m",
                        ref=10,
                        desc="Full plasma height")
        self.add_output("S_c",
                        units="m**2",
                        ref=10,
                        desc="Poloidal cross-section area")
        self.add_output("S", units="m**2", ref=100, desc="Surface area")
        self.add_output("V", units="m**3", ref=100, desc="Volume")
        self.add_output("L_pol",
                        units="m",
                        desc="Poloidal circumference",
                        lower=0,
                        ref=10)
        self.add_output("R_in",
                        units="m",
                        lower=0,
                        desc="Inner radius of plasma at midplane")
        self.add_output("R_out",
                        units="m",
                        lower=0,
                        desc="outer radius of plasma at midplane")
        self.add_output("w07", desc="Sauter 70% width parameter")
        self.add_output("κa", lower=0, desc="effective elongation")
        self.add_output("dR_dθ", units="m", copy_shape="θ")
        self.add_output("dZ_dθ", units="m", copy_shape="θ")

    def compute(self, inputs, outputs):
        outputs["Z0"] = 0

        R0 = inputs["R0"]
        a = inputs["a"]
        A = inputs["A"]
        κ = inputs["κ"]
        δ = inputs["δ"]
        ξ = inputs["ξ"]

        θ = inputs["θ"]

        # Sauter equation (C.2)
        # but using the 'alternate expression for a quadratic root'
        # 2c / (-b + √(b² - 4 a c))
        θ07 = np.arcsin(0.7) - (2 * ξ) / (1 + (1 + 8 * ξ**2)**(1 / 2))

        # Sauter equation (C.5)
        wanal_07 = cos(θ07 - ξ * sin(2 * θ07) / (0.51)**(1/2))  \
            * (1 - 0.49 * δ**2 / 2)
        w07 = wanal_07

        # Sauter equation (25), also (36)
        L_p = (2 * pi * a * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
               (1 + 0.2 * (w07 - 1)))

        outputs["L_pol"] = L_p

        # Equation (1)
        R = R0 + a * cos(θ + δ * sin(θ) - ξ * sin(2 * θ))
        # Equation (2)
        Z = κ * a * sin(θ + ξ * sin(2 * θ))

        b = κ * a
        outputs["b"] = b
        outputs["full_plasma_height"] = 2 * b

        outputs["R"] = R
        outputs["Z"] = Z

        ε = 1 / A
        outputs["ε"] = ε

        # Equation (26), also (36)
        Ap = (2 * pi * R0) * (1 - 0.32 * δ * ε) * L_p

        # Equation (29), also (36); here Sφ is called S_c
        S_c = (pi * a**2 * κ) * (1 + 0.52 * (w07 - 1))
        outputs["κa"] = S_c / (pi * a**2)

        # Equation (28), also (36)
        V = (2 * pi * R0) * (1 - 0.25 * δ * ε) * S_c
        outputs["S_c"] = S_c
        outputs["S"] = Ap
        outputs["V"] = V

        outputs["R_in"] = R0 - a
        outputs["R_out"] = R0 + a
        outputs["w07"] = w07

        dR_dθ = -a * (1 + δ * cos(θ) - 2 * ξ *
                      cos(2 * θ)) * sin(θ + δ * sin(θ) - ξ * sin(2 * θ))
        dZ_dθ = a * κ * (1 + 2 * ξ * cos(2 * θ)) * cos(θ + ξ * sin(2 * θ))
        outputs["dR_dθ"] = dR_dθ
        outputs["dZ_dθ"] = dZ_dθ

    def setup_partials(self):
        size = self._get_var_meta("θ", "size")
        self.declare_partials("ε", ["A"])
        self.declare_partials("full_plasma_height", ["a", "κ"])

        self.declare_partials(["R_in", "R_out"], "R0", val=1)
        self.declare_partials("R_in", "a", val=-1)
        self.declare_partials("R_out", "a", val=1)

        self.declare_partials("b", ["a", "κ"])
        self.declare_partials("w07", ["δ", "ξ"])

        self.declare_partials("L_pol", ["a", "κ", "δ", "ξ"])
        self.declare_partials("S", ["R0", "A", "a", "κ", "δ", "ξ"])
        self.declare_partials("S_c", ["a", "κ", "δ", "ξ"])
        self.declare_partials("V", ["R0", "A", "a", "κ", "δ", "ξ"])

        self.declare_partials("κa", ["a", "κ", "δ", "ξ"])

        self.declare_partials("R", ["R0", "a", "δ", "ξ"])
        self.declare_partials("R", ["θ"], rows=range(size), cols=range(size))

        self.declare_partials("Z", ["a", "κ", "ξ"])
        self.declare_partials("Z", ["θ"], rows=range(size), cols=range(size))
        self.declare_partials("dR_dθ", ["a", "ξ", "δ"])
        self.declare_partials("dR_dθ", ["θ"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("dZ_dθ", ["a", "ξ", "κ"])
        self.declare_partials("dZ_dθ", ["θ"],
                              rows=range(size),
                              cols=range(size))

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
        ε = 1 / A
        J["ε", "A"] = -1 / A**2

        J["b", "κ"] = a
        J["b", "a"] = κ

        J["full_plasma_height", "κ"] = 2 * J["b", "κ"]
        J["full_plasma_height", "a"] = 2 * J["b", "a"]

        # Sauter equation (C.2)
        # but using the 'alternate expression for a quadratic root'
        # 2c / (-b + √(b² - 4 a c))
        θ07 = np.arcsin(0.7) - (2 * ξ) / (1 + (1 + 8 * ξ**2)**(1 / 2))
        # Sauter equation (C.5)
        wanal_07 = cos(θ07 - ξ * sin(2 * θ07) / (0.51)**(1/2))  \
            * (1 - 0.49 * δ**2 / 2)
        w07 = wanal_07

        dθ_dξ = -2 / (1 + 8 * ξ**2 + (1 + 8 * ξ**2)**(1 / 2))

        dw07_dξ = (sin(2 * θ07) * sin(θ07 - ξ * sin(2 * θ07) /
                                      (0.51)**(1 / 2))) * (
                                          1 - 0.49 * δ**2 / 2) / 0.51**(1 / 2)
        dw07_dθ07 = -(
            (1 - 2 * ξ * cos(2 * θ07) / 0.51**(1 / 2)) *
            sin(θ07 - ξ * sin(2 * θ07) / 0.51**0.5)) * (1 - 0.49 * δ**2 / 2)

        dw07_dδ = -0.49 * δ * cos(θ07 - ξ * sin(2 * θ07) / 0.51**(1 / 2))
        J["w07", "δ"] = dw07_dδ
        J["w07", "ξ"] = dw07_dξ + dw07_dθ07 * dθ_dξ

        L_p = (2 * pi * a * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
               (1 + 0.2 * (w07 - 1)))
        # partials
        dLp_da = (2 * pi * (1 + 0.55 * (κ - 1)) * (1 + 0.08 * δ**2) *
                  (1 + 0.2 * (w07 - 1)))
        dLp_dκ = 11 / 20 * (2 * pi * a * (1 + 0.08 * δ**2) * (1 + 0.2 *
                                                              (w07 - 1)))
        dLp_dδ = 0.16 * δ * (2 * pi * a * (1 + 0.55 * (κ - 1)) * (1 + 0.2 *
                                                                  (w07 - 1)))
        dLp_dw07 = 0.2 * (2 * pi * a * (1 + 0.08 * δ**2) * (1 + 0.55 *
                                                            (κ - 1)))

        J["L_pol", "a"] = dLp_da
        J["L_pol", "κ"] = dLp_dκ
        J["L_pol", "δ"] = dLp_dδ + dLp_dw07 * J["w07", "δ"]
        J["L_pol", "ξ"] = dLp_dw07 * J["w07", "ξ"]

        # partials
        dAp_dR0 = 2 * pi * (1 - 0.32 * δ * ε) * L_p
        dAp_dLp = 2 * pi * R0 * (1 - 0.32 * δ * ε)
        dAp_dA = 2 * pi * R0 * (+0.32 * δ * 1 / A**2) * L_p
        dAp_dδ = 2 * pi * R0 * (-0.32 * ε) * L_p

        J["S", "A"] = dAp_dA
        J["S", "R0"] = dAp_dR0
        J["S", "a"] = dAp_dLp * J["L_pol", "a"]
        J["S", "κ"] = dAp_dLp * J["L_pol", "κ"]
        J["S", "δ"] = dAp_dδ + dAp_dLp * J["L_pol", "δ"]
        J["S", "ξ"] = dAp_dLp * J["L_pol", "ξ"]

        S_c = (pi * a**2 * κ) * (1 + 0.52 * (w07 - 1))
        dSc_dw07 = (pi * a**2 * κ) * 0.52

        dSc_da = 2 * pi * a * κ * (1 + 0.52 * (w07 - 1))
        J["S_c", "a"] = dSc_da
        J["S_c", "κ"] = (pi * a**2) * (1 + 0.52 * (w07 - 1))
        J["S_c", "δ"] = dSc_dw07 * J["w07", "δ"]
        J["S_c", "ξ"] = dSc_dw07 * J["w07", "ξ"]

        dκa_da = -2 * S_c / (pi * a**3)
        dκa_dSc = 1 / (pi * a**2)
        J["κa", "a"] = dκa_da + dκa_dSc * J["S_c", "a"]
        J["κa", "κ"] = dκa_dSc * J["S_c", "κ"]
        J["κa", "δ"] = dκa_dSc * J["S_c", "δ"]
        J["κa", "ξ"] = dκa_dSc * J["S_c", "ξ"]

        dV_dR0 = (2 * pi) * (1 - 0.25 * δ * 1 / A) * S_c
        dV_dA = (2 * pi * R0) * (-0.25 * δ * -1 / A**2) * S_c
        dV_dδ = (2 * pi * R0) * (-0.25 * 1 / A) * S_c
        dV_dSc = (2 * pi * R0) * (1 - 0.25 * δ * 1 / A)
        J["V", "R0"] = dV_dR0
        J["V", "A"] = dV_dA
        J["V", "a"] = (2 * pi * R0) * (1 - 0.25 * δ * ε) * dSc_da
        J["V", "κ"] = dV_dSc * J["S_c", "κ"]
        J["V", "δ"] = dV_dδ + dV_dSc * J["S_c", "δ"]
        J["V", "ξ"] = dV_dSc * J["S_c", "ξ"]

        J["R", "a"] = cos(θ + δ * sin(θ) - ξ * sin(2 * θ))
        J["R", "R0"] = 1
        J["R", "δ"] = -a * sin(θ) * sin(θ + δ * sin(θ) - ξ * sin(2 * θ))
        J["R", "ξ"] = a * sin(2 * θ) * sin(θ + δ * sin(θ) - ξ * sin(2 * θ))

        dR_dθ_da = -1 * (1 + δ * cos(θ) - 2 * ξ *
                         cos(2 * θ)) * sin(θ + δ * sin(θ) - ξ * sin(2 * θ))
        dR_dθ = a * dR_dθ_da

        J["R", "θ"] = dR_dθ

        J["Z", "a"] = κ * sin(θ + ξ * sin(2 * θ))
        J["Z", "κ"] = a * sin(θ + ξ * sin(2 * θ))
        J["Z", "ξ"] = a * κ * cos(θ + ξ * sin(2 * θ)) * sin(2 * θ)
        dZ_dθ_da = κ * (1 + 2 * ξ * cos(2 * θ)) * cos(θ + ξ * sin(2 * θ))
        dZ_dθ = a * dZ_dθ_da
        J["Z", "θ"] = dZ_dθ

        J["dR_dθ", "a"] = dR_dθ_da
        J["dR_dθ", "δ"] = -a * (1 + δ * cos(θ) - 2 * ξ * cos(2 * θ)) * cos(
            θ + δ * sin(θ) - ξ * sin(2 * θ)) * sin(θ) - a * cos(θ) * sin(
                θ + δ * sin(θ) - ξ * sin(2 * θ))
        J["dR_dθ", "ξ"] = a * (1 + δ * cos(θ) - 2 * ξ * cos(2 * θ)) * cos(
            θ + δ * sin(θ) - ξ * sin(2 * θ)) * sin(2 * θ) + 2 * a * cos(
                2 * θ) * sin(θ + δ * sin(θ) - ξ * sin(2 * θ))
        d2R_dθ2 = -a * (1 + δ * cos(θ) - 2 * ξ * cos(2 * θ))**2 * cos(
            θ + (δ - 2 * ξ * cos(θ)) * sin(θ)) + a * (δ * sin(θ) - 4 * ξ * sin(
                2 * θ)) * sin(θ + (δ - 2 * ξ * cos(θ)) * sin(θ))
        J["dR_dθ", "θ"] = d2R_dθ2

        J["dZ_dθ", "a"] = dZ_dθ_da
        J["dZ_dθ",
          "κ"] = a * (1 + 2 * ξ * cos(2 * θ)) * cos(θ + ξ * sin(2 * θ))
        J["dZ_dθ", "ξ"] = a * κ * (
            2 * cos(2 * θ) * cos(θ + ξ * sin(2 * θ)) -
            (1 + 2 * ξ * cos(2 * θ)) * sin(2 * θ) * sin(θ + ξ * sin(2 * θ)))
        d2Z_dθ2 = a * κ * (
            -4 * ξ * cos(θ + ξ * sin(2 * θ)) * sin(2 * θ) -
            (1 + 2 * ξ * cos(2 * θ))**2 * sin(θ + ξ * sin(2 * θ)))
        J["dZ_dθ", "θ"] = d2Z_dθ2

    def plot(self, ax=None, **kwargs):
        ax.plot(self.get_val('R0'), self.get_val('Z0'), marker='x', **kwargs)
        ax.plot(self.get_val('R'), self.get_val('Z'), **kwargs)


class SauterPlasmaGeometryMarginalKappa(om.Group):
    r"""Sauter's general plasma shape and blanket with automatic elongation.

    Uses :code:`MenardKappaScaling`.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("maxkappa",
                           MenardKappaScaling(config=config),
                           promotes_inputs=["A"],
                           promotes_outputs=["κ"])
        self.add_subsystem(
            "geom",
            SauterGeometry(),
            promotes_inputs=["R0", "a", "A", "κ", "δ", "ξ", "θ"],
            promotes_outputs=["*"])


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()

    θ = np.linspace(0, 2 * pi, 10, endpoint=False)

    prob.model.add_subsystem("ivc",
                             om.IndepVarComp("θ", val=θ),
                             promotes_outputs=["*"])

    sg = SauterPlasmaGeometryMarginalKappa(config=uc)
    prob.model.add_subsystem("spg", sg, promotes_inputs=["*"])

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('A', 1.6)
    prob.set_val('a', 1.875)
    prob.set_val('κ', 2.7)
    prob.set_val('δ', 0.5)
    prob.set_val('ξ', 0.1)

    prob.run_driver()
    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)
