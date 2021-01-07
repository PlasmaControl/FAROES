import openmdao.api as om
import faroes.util as util
import numpy as np

# class PlasmaBetaNTotal(om.ExplicitComponent):
#     r"""Total β_N from a scaling law
#
#     Inputs
#     ------
#     A : float
#         Aspect ratio
#
#     Outputs
#     -------
#     beta_N : float
#         β_N, the normalized beta
#
#     beta_N total : float
#         total β_N, the total normalized beta
#
#     Reference
#     ---------
#     Physics of Plasmas 11, 639 (2004);
#     https://doi.org/10.1063/1.1640623
#
#     No-wall limit, with 50% bootstrap fraction
#     """
#     def initialize(self):
#         self.options.declare('config', default=None)
#
#     def setup(self):
#         if self.options['config'] is not None:
#             self.config = self.options['config']
#             ac = self.config.accessor(['fits'])
#             self.β_ε_scaling_constants = ac(
#                 ["no-wall β_N scaling with ε", "constants"])
#             self.β_N_multiplier = ac(["β_N multiplier"])
#
#         self.add_input("A", desc="Aspect Ratio")
#         self.add_output("beta_N", desc="Normalized beta")
#         self.add_output("beta_N total", desc="Normalized beta, total")
#
#     def beta_N_scaling(self, A):
#         """Estimated β_N from A
#
#         Parameters
#         ----------
#         A: float
#             aspect ratio of the plasma
#
#         """
#         const = self.β_ε_scaling_constants
#         b = const[0]
#         c = const[1]
#         d = const[2]
#         return b + c / (A**d)
#
#     def compute(self, inputs, outputs):
#         A = inputs["A"]
#         β_N = self.beta_N_scaling(A)
#         outputs["beta_N"] = β_N
#         outputs["beta_N total"] = β_N * self.β_N_multiplier
#
#     def setup_partials(self):
#         self.declare_partials('*', '*', method='cs')


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
        \delta &= \frac{\delta_\mathrm{top} + \delta_\mathrm{bot}}{2}
        \delta_\mathrm{top} = \frac{R_0 - R(Z=Z_\mathrm{max})}{a}
        \delta_\mathrm{bot} = \frac{R_0 - R(Z=Z_\mathrm{min})}{a}

    Inputs
    ------
    R0 : float
        m, major radius
    a : float
        m, minor radius
    κ : float
        elongation
    δ : float
        triangularity
    ξ : float
        related to the plasma squareness
    nθ : int
        number of poloidal divisions in plasma boundary (default = 180)

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
    # κa : float
    #     effective elongation, S_c / (π a^2),
    #     where S_c is the plasma cross-sectional area
    #     same as κ for this elliptical plasma.
    full plasma height : float
        m, twice b
    S : float
        m**2, surface area
    V : float
        m**3, volume of the elliptical torus
    R_min : float
        m, innermost plasma radius at midplane
    R_max : float
        m, outermost plasma radius at midplane

    Reference
    ---------
    Fusion Engineering and Design 112, 633-645 (2016);
    http://dx.doi.org/10.1016/j.fusengdes.2016.04.033

    """
    def initialize(self):
        self.options.declare('config', default=None)
        self.options.declare('nθ', types=int, default=180,
                        desc="Number of poloidal divisions in plasma boundary")

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']

        nθ = self.options["nθ"]

        self.add_input("R0", units='m', desc="Major radius")
        self.add_input("a", units='m', desc="Minor radius")
        self.add_input("κ", desc="Elongation")
        self.add_input("δ", desc="Triangularity")
        self.add_input("ξ", desc="Related to the plasma squareness")

        self.add_output("R", units='m', shape=(nθ),
                        desc="Radial locations of plasma boundary")
        self.add_output("Z", units='m', shape=(nθ),
                        desc="Vertical locations of plasma boundary")
        self.add_output("Z0", units='m',
                        desc="vertical location of magnetic axis")
        self.add_output("b", units='m', desc="Minor radius height")
        self.add_output("ε", desc="Inverse aspect ratio")
        # self.add_output("κa", desc="Effective elongation")
        self.add_output("full_plasma_height",
                        units='m',
                        desc="Full plasma height")
        self.add_output("S",
                        units='m**2',
                        desc="Surface area")
        self.add_output("V", units='m**3', desc="Volume")
        self.add_output("R_min",
                        units='m',
                        desc="Inner radius of plasma at midplane")
        self.add_output("R_max",
                        units='m',
                        desc="outer radius of plasma at midplane")

    def compute(self, inputs, outputs):
        R0 = inputs["R0"]
        a = inputs["a"]
        κ = inputs["κ"]
        δ = inputs["δ"]
        ξ = inputs["ξ"]

        nθ = self.options["nθ"]
        θ = np.linspace(0, 2 * np.pi, nθ)

        R = R0 + a * np.cos(θ + δ * np.sin(θ) - ξ * np.sin(2 * θ))
        Z = κ * a * np.sin(θ + ξ * np.sin(2 * θ))

        b = max(Z) - min(Z)
        #S = util.torus_surface_area(R0, a, b)
        #V = util.torus_volume(R0, a, b)

        outputs["R"] = R
        outputs["Z"] = Z

        outputs["Z0"] = 0
        outputs["b"] = b
        outputs["ε"] = a / R0
        outputs["full_plasma_height"] = 2 * b
        #outputs["S"] = S
        #outputs["V"] = V

        outputs["R_min"] = R0 - a
        outputs["R_max"] = R0 + a

    # def setup_partials(self):
    #     self.declare_partials('*', '*', method='cs')
    #     self.declare_partials('R_min', ['R0', 'A'])
    #     self.declare_partials('R_max', ['R0', 'A'])
    #
    # def compute_partials(self, inputs, J):
    #     A = inputs['A']
    #     R0 = inputs['R0']
    #     J["R_min", "R0"] = 1 - 1 / A
    #     J["R_min", "A"] = R0 / A**2
    #     J["R_max", "R0"] = 1 + 1 / A
    #     J["R_max", "A"] = -R0 / A**2


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = SauterGeometry()

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('a', 1, 'm')
    prob.set_val('κ', 0.5)
    prob.set_val('δ', 0.5)
    prob.set_val('ξ', 0.)

    prob.run_driver()

    # test by plotting
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(prob.get_val('R'),prob.get_val('Z'))
    # plt.show()
