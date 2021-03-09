from faroes.configurator import UserConfigurator
import faroes.util as util

import openmdao.api as om
from math import pi as π


class MenardKappaScaling(om.ExplicitComponent):
    r"""

    Elongation :math:`\kappa` is determined by a "marginal scaling"
    from Jon Menard:

    .. math:: \kappa = 0.95 (1.9 + 1.9 / (A^{1.4})).

    An factor ":math:`\kappa` area fraction" from the configuration file
    multiplies the above "marginal :math:`\kappa`"
    to generate :math:`\kappa_a`. The latter is used for computations
    of the plasma cross-sectional area and volume.

    Inputs
    ------
    A : float
        Aspect ratio

    Outputs
    -------
    κ : float
        Elongation
    κa : float
        Effective elongation

    Notes
    -----
    The fit coefficients are loaded from the configuration tree.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['fits'])
            self.kappa_multiplier = ac(["κ multiplier"])
            self.κ_area_frac = ac(["κ area fraction"])
            self.κ_ε_scaling_constants = ac(
                ["marginal κ-ε scaling", "constants"])

        self.add_input("A", desc="Aspect Ratio")
        self.add_output("κ", lower=0, ref=2, desc="Elongation")
        self.add_output("κa", lower=0, ref=2, desc="Effective elongation")

    def marginal_kappa_epsilon_scaling(self, aspect_ratio):
        """
        marginal kappa(epsilon) scaling

        Parameters
        ----------
        aspect_ratio: float

        References
        ----------
        Physics of Plasmas 11, 639 (2004);
        https://doi.org/10.1063/1.1640623
        "NSTX, FNSF-ST, DIII-D, EAST"
        I guess this is how kappa scales with epsilon.

        I'm not sure where it is in the paper, though.

        [T73 from the spreadsheet]

        b12 + c12/(t4^d12)

        Values are from the "Scaling Parameters" sheet
        """
        constants = self.κ_ε_scaling_constants
        b = constants[0]
        c = constants[1]
        d = constants[2]
        return b + c / (aspect_ratio**d)

    def compute(self, inputs, outputs):
        A = inputs["A"]

        if A <= 1:
            raise om.AnalysisError(f"Aspect ratio A ={A} < 1")


        κ = self.kappa_multiplier * self.marginal_kappa_epsilon_scaling(A)
        outputs["κ"] = κ
        outputs["κa"] = κ * self.κ_area_frac

    def setup_partials(self):
        self.declare_partials(["κ", "κa"], "A")

    def compute_partials(self, inputs, J):
        constants = self.κ_ε_scaling_constants
        sp_c12 = constants[1]
        sp_d12 = constants[2]
        A = inputs["A"]
        J["κ",
          "A"] = -self.kappa_multiplier * sp_c12 * sp_d12 * A**(-sp_d12 - 1)
        J["κa", "A"] = J["κ", "A"] * self.κ_area_frac


class EllipseLikeGeometry(om.ExplicitComponent):
    r"""Describes an elliptical or ellipse-like plasma.

    .. image :: images/ellipticalplasmageometry.png

    .. math::

        a &= R_0 / A \\
        \delta &= 0

    :math:`\kappa_a` is an 'effective elongation' which is the same
    as :math:`\kappa` for a perfect ellipse plasma, but can be smaller
    to model a more realistic diverted plasma shape or volume.
    The "ellipse-like plasma" has :math:`\kappa_a \neq \kappa`.

    Other properties are determined using standard geometry formulas:

    .. math::

        b &= \kappa a \\
        \epsilon &= 1 / A \\
        \textrm{full plasma height} &= 2 b \\
        L_\mathrm{pol} &= \textrm{ellipse_perimeter(a, b)} \\
        \textrm{surface area} &= 2 \pi R * \textrm{ellipse_perimeter(a, b)} \\
        R_\mathrm{min} &= R - a \\
        R_\mathrm{max} &= R + a \\
        V &= 2 \pi R * \pi a^2 \kappa_a \\
        S_c &= \pi a^2 \kappa_a .

    where :math:`L_\mathrm{pol}` is the poloidal circumference and :math:`S_c`
    is the poloidal cross-section area.

    Inputs
    ------
    R0 : float
        m, major radius
    A : float
        Aspect ratio
    κ : float
        Elongation
    κa : float
        effective elongation, :math:`S_c / (\pi a^2)`,
        where :math:`S_c` is the plasma cross-sectional area.
        same as :math:`\kappa` for this elliptical plasma.

    Outputs
    -------
    a : float
        m, minor radius
    R_min : float
        m, innermost plasma radius at midplane
    R_max : float
        m, outermost plasma radius at midplane
    b : float
        m, minor radius in vertical direction
    δ : float
        triangularity: is always zero
    ε : float
        inverse aspect ratio
    full plasma height : float
        m, Twice b
    surface area : float
        m**2, surface area
    L_pol : float
        m, Poloidal circumference
    L_pol_simple : float
        m, Simplified poloidal circumference for testing.
            Uses a simple ellipse approximation.
    S_c : float
        m**2, Poloidal cross section area
    V : float
        m**3, Plasma volume
    """
    def setup(self):
        self.add_input("R0", units='m', desc="Major radius")
        self.add_input("A", desc="Aspect Ratio")
        self.add_input("κ", desc="Elongation")
        self.add_input("κa", desc="Effective elongation")

        self.add_output("a", units='m', desc="Minor radius")
        self.add_output("b", units='m', desc="Minor radius height")
        self.add_output("ε", desc="Inverse aspect ratio")
        self.add_output("δ", desc="Triangularity")
        self.add_output("full_plasma_height",
                        units='m',
                        desc="Top to bottom of the ellipse")
        self.add_output("surface area",
                        units='m**2',
                        lower=0,
                        ref=100,
                        desc="Surface area of the elliptical plasma")
        self.add_output("V", units='m**3', desc="Volume", lower=0, ref=100)
        self.add_output("S_c",
                        units='m**2',
                        desc="Cross-sectional area",
                        lower=0,
                        ref=20)
        self.add_output("L_pol",
                        units="m",
                        desc="Poloidal circumference",
                        lower=0,
                        ref=10)
        self.add_output("L_pol_simple",
                        units="m",
                        lower=0,
                        ref=10,
                        desc="Simplified poloidal circumference for testing")
        self.add_output("R_min",
                        units='m',
                        desc="Inner radius of plasma at midplane")
        self.add_output("R_max",
                        units='m',
                        desc="outer radius of plasma at midplane")

    def compute(self, inputs, outputs):
        outputs["δ"] = 0  # ellipse-like plasma approximation
        R0 = inputs["R0"]
        A = inputs["A"]
        κ = inputs["κ"]
        κa = inputs["κa"]
        a = R0 / A

        b = κ * a
        L_pol = util.ellipse_perimeter(a[0], b[0])
        L_pol_simple = util.ellipse_perimeter_simple(a[0], b[0])
        sa = 2 * π * R0 * L_pol

        outputs["a"] = a
        outputs["ε"] = 1 / A
        outputs["b"] = b
        outputs["full_plasma_height"] = 2 * b
        outputs["surface area"] = sa

        outputs["R_min"] = R0 - a
        outputs["R_max"] = R0 + a
        outputs["L_pol"] = L_pol
        outputs["L_pol_simple"] = L_pol_simple

        outputs["S_c"] = π * a**2 * κa
        V = 2 * π**2 * κa * a**2 * R0
        outputs["V"] = V

    def setup_partials(self):
        self.declare_partials("a", ["R0", "A"], method="exact")
        self.declare_partials('R_min', ['R0', 'A'], method="exact")
        self.declare_partials('R_max', ['R0', 'A'], method="exact")

        self.declare_partials("b", ["R0", "A", "κ"], method="exact")
        self.declare_partials("ε", ["A"], method="exact")
        self.declare_partials("full_plasma_height", ["R0", "A", "κ"],
                              method="exact")
        self.declare_partials("L_pol", ["R0", "A", "κ"], method="exact")
        self.declare_partials("L_pol_simple", ["R0", "A", "κ"], method="exact")
        self.declare_partials("surface area", ["R0", "A", "κ"], method="exact")

        self.declare_partials("V", ["R0", "A", "κa"], method="exact")
        self.declare_partials("S_c", ["R0", "A", "κa"], method="exact")

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        R0 = inputs["R0"]
        κ = inputs["κ"]
        κa = inputs["κa"]
        a = R0 / A
        b = κ * a
        L_pol = util.ellipse_perimeter(a[0], b[0])

        J["R_min", "R0"] = 1 - 1 / A
        J["R_min", "A"] = R0 / A**2
        J["R_max", "R0"] = 1 + 1 / A
        J["R_max", "A"] = -R0 / A**2

        J["a", "R0"] = 1 / A
        J["a", "A"] = -R0 / A**2
        J["b", "R0"] = κ / A
        J["b", "A"] = -κ * R0 / A**2
        J["b", "κ"] = R0 / A
        J["full_plasma_height", "R0"] = 2 * J["b", "R0"]
        J["full_plasma_height", "A"] = 2 * J["b", "A"]
        J["full_plasma_height", "κ"] = 2 * J["b", "κ"]
        J["ε", "A"] = -1 / A**2

        dL_pol = util.ellipse_perimeter_derivatives(a[0], b[0])
        dL_pol_da = dL_pol["a"]
        dL_pol_db = dL_pol["b"]
        J["L_pol", "A"] = dL_pol_da * J["a", "A"] + dL_pol_db * J["b", "A"]
        J["L_pol", "R0"] = dL_pol_da * J["a", "R0"] + dL_pol_db * J["b", "R0"]
        J["L_pol", "κ"] = dL_pol_db * J["b", "κ"]

        dL_pols = util.ellipse_perimeter_simple_derivatives(a[0], b[0])
        dL_pols_da = dL_pols["a"]
        dL_pols_db = dL_pols["b"]
        J["L_pol_simple",
          "A"] = dL_pols_da * J["a", "A"] + dL_pols_db * J["b", "A"]
        J["L_pol_simple",
          "R0"] = dL_pols_da * J["a", "R0"] + dL_pols_db * J["b", "R0"]
        J["L_pol_simple", "κ"] = dL_pols_db * J["b", "κ"]

        J["surface area", "A"] = 2 * π * R0 * J["L_pol", "A"]
        J["surface area", "R0"] = 2 * π * R0 * J["L_pol", "R0"] + 2 * π * L_pol
        J["surface area", "κ"] = 2 * π * R0 * J["L_pol", "κ"]

        J["V", "R0"] = 2 * π**2 * κa * 3 * R0**2 / A**2
        J["V", "A"] = 2 * π**2 * κa * -2 * R0**3 / A**3
        J["V", "κa"] = 2 * π**2 * a**2 * R0
        J["S_c", "R0"] = 2 * π * R0 * κa / A**2
        J["S_c", "A"] = -2 * π * R0**2 * κa / A**3
        J["S_c", "κa"] = π * a**2


class EllipticalPlasmaGeometry(om.Group):
    r"""Perfect ellipse plasma.

    Fixes :math:`\kappa_a = \kappa`.

    Otherwise behaves like :code:`EllipseLikePlasma`.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("kappa",
                           MenardKappaScaling(config=config),
                           promotes_inputs=["A"],
                           promotes_outputs=["κ"])
        # enforce equality
        self.add_subsystem("passthrough",
                           om.ExecComp("kappaa = kappa"),
                           promotes_inputs=[("kappa", "κ")],
                           promotes_outputs=[("kappaa", "κa")])

        self.add_subsystem("geom",
                           EllipseLikeGeometry(),
                           promotes_inputs=["R0", "A", "κ", "κa"],
                           promotes_outputs=["*"])


class MenardPlasmaGeometry(om.Group):
    r"""Not-quite-ellipse plasma shape.

    Loads "κ area fraction" from a configuration file
    and has :math:`\kappa_a \neq \kappa`.

    Otherwise behaves like :code:`EllipseLikePlasma`.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("kappa",
                           MenardKappaScaling(config=config),
                           promotes_inputs=["A"],
                           promotes_outputs=["κ", "κa"])
        self.add_subsystem("geom",
                           EllipseLikeGeometry(),
                           promotes_inputs=["R0", "A", "κ", "κa"],
                           promotes_outputs=["*"])


if __name__ == "__main__":
    uc = UserConfigurator()
    prob = om.Problem()

    prob.model = EllipticalPlasmaGeometry(config=uc)

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('A', 1.6)
    prob.set_val('κ', 2.7)
    prob.set_val('κa', 2.7)

    prob.run_driver()
    prob.model.list_inputs()
    prob.model.list_outputs()
