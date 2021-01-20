import openmdao.api as om
import faroes.util as util


class PlasmaBetaNTotal(om.ExplicitComponent):
    r"""Total β_N from a scaling law

    Inputs
    ------
    A : float
        Aspect ratio

    Outputs
    -------
    beta_N : float
        β_N, the normalized beta

    beta_N total : float
        total β_N, the total normalized beta

    Reference
    ---------
    Physics of Plasmas 11, 639 (2004);
    https://doi.org/10.1063/1.1640623

    No-wall limit, with 50% bootstrap fraction
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['fits'])
            self.β_ε_scaling_constants = ac(
                ["no-wall β_N scaling with ε", "constants"])
            self.β_N_multiplier = ac(["β_N multiplier"])

        self.add_input("A", desc="Aspect Ratio")
        self.add_output("beta_N", desc="Normalized beta")
        self.add_output("beta_N total", desc="Normalized beta, total")

    def beta_N_scaling(self, A):
        """Estimated β_N from A

        Parameters
        ----------
        A: float
            aspect ratio of the plasma

        """
        const = self.β_ε_scaling_constants
        b = const[0]
        c = const[1]
        d = const[2]
        return b + c / (A**d)

    def compute(self, inputs, outputs):
        A = inputs["A"]
        β_N = self.beta_N_scaling(A)
        outputs["beta_N"] = β_N
        outputs["beta_N total"] = β_N * self.β_N_multiplier

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

class KappaScaling(om.ExplicitComponent):
    """
    Inputs
    ------
    A : float
        Aspect ratio

    Outputs
    -------
    κ : float
        Elongation

    κ is determined by a "marginal kappa scaling"
    from Jon Menard:

    .. math::

       \kappa = 0.95 (1.9 + 1.9 / (A^{1.4})).

    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['fits'])
            self.kappa_multiplier = ac(["κ multiplier"])
            self.κ_ε_scaling_constants = ac(
                ["marginal κ-ε scaling", "constants"])

        self.add_input("A", desc="Aspect Ratio")
        self.add_output("κ", lower=0, ref=2, desc="Elongation")

    def marginal_kappa_epsilon_scaling(self, t4_aspect_ratio):
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
        sp_b12 = constants[0]
        sp_c12 = constants[1]
        sp_d12 = constants[2]
        return sp_b12 + sp_c12 / (t4_aspect_ratio**sp_d12)

    def compute(self, inputs, outputs):
        A = inputs["A"]
        κ = self.kappa_multiplier * self.marginal_kappa_epsilon_scaling(A)
        outputs["κ"] = κ

    def setup_partials(self):
        self.declare_partials("κ", "A")

    def compute_partials(self, inputs, J):
        constants = self.κ_ε_scaling_constants
        sp_c12 = constants[1]
        sp_d12 = constants[2]
        A = inputs["A"]
        J["κ", "A"] = -self.kappa_multiplier * sp_c12 * sp_d12 * A**(-sp_d12 -1)


class PlasmaGeometry(om.ExplicitComponent):
    r"""Describes an elliptical plasma.

    .. image :: images/ellipticalplasmageometry.png

    .. math::

        a &= R_0 / A \\
        \delta &= 0

    κ is determined by a "marginal kappa scaling"
    from Jon Menard:

    .. math:: \kappa = 0.95 (1.9 + 1.9 / (A^{1.4})).

    κa an 'effective elongation' which is the same as κ,
    since this is an elliptical plasma:

    .. math::

        {\kappa}a \equiv \kappa.

    Other properties are determined using standard geometry formulas:

    .. math::

        b &= \kappa * a \\
        \epsilon &= 1 / A \\
        \textrm{full plasma height} &= 2 b \\
        \textrm{surface area} &= 2 \pi R * \textrm{ellipse_perimeter(a, b)} \\
        V &= 2 \pi R * \pi a b \\
        R_\mathrm{min} &= R - a \\
        R_\mathrm{max} &= R + a.

    Inputs
    ------
    R0 : float
        m, major radius
    A : float
        Aspect ratio

    Outputs
    -------
    a : float
        m, minor radius
    b : float
        m, minor radius in vertical direction
    δ : float
        triangularity: is always zero
    ε : float
        inverse aspect ratio
    κ : float
        elongation
    κa : float
        effective elongation, S_c / (π a^2),
        where S_c is the plasma cross-sectional area
        same as κ for this elliptical plasma.
    full plasma height : float
        m, Twice b
    surface area : float
        m**2, surface area
    V : float
        m**3, volume of the elliptical torus
    L_pol : float
        m, Poloidal circumference
    R_min : float
        m, innermost plasma radius at midplane
    R_max : float
        m, outermost plasma radius at midplane
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['fits'])
            self.kappa_multiplier = ac(["κ multiplier"])
            self.κ_ε_scaling_constants = ac(
                ["marginal κ-ε scaling", "constants"])

        self.add_input("R0", units='m', desc="Major radius")
        self.add_input("A", desc="Aspect Ratio")

        self.add_output("a", units='m', desc="Minor radius")
        self.add_output("b", units='m', desc="Minor radius height")
        self.add_output("ε", desc="Inverse aspect ratio")
        self.add_output("κ", desc="Elongation")
        self.add_output("κa", desc="Elongation")
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
        self.add_output("L_pol", units="m", desc="Poloidal circumference", lower=0, ref=10)
        self.add_output("R_min",
                        units='m',
                        desc="Inner radius of plasma at midplane")
        self.add_output("R_max",
                        units='m',
                        desc="outer radius of plasma at midplane")

    def compute(self, inputs, outputs):
        outputs["δ"] = 0  # elliptical plasma approximation
        R0 = inputs["R0"]
        A = inputs["A"]
        a = R0 / A

        κ = self.kappa_multiplier * self.marginal_kappa_epsilon_scaling(A)
        b = κ * a
        sa = util.torus_surface_area(R0, a, b)
        V = util.torus_volume(R0, a, b)
        outputs["κ"] = κ
        outputs["κa"] = κ
        outputs["a"] = a
        outputs["ε"] = 1 / A
        outputs["b"] = b
        outputs["full_plasma_height"] = 2 * b
        outputs["surface area"] = sa
        outputs["V"] = V

        outputs["R_min"] = R0 - a
        outputs["R_max"] = R0 + a
        outputs["L_pol"] = util.ellipse_perimeter(a[0], b[0])

    def marginal_kappa_epsilon_scaling(self, t4_aspect_ratio):
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
        sp_b12 = constants[0]
        sp_c12 = constants[1]
        sp_d12 = constants[2]
        return sp_b12 + sp_c12 / (t4_aspect_ratio**sp_d12)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
        self.declare_partials('R_min', ['R0', 'A'], method="exact")
        self.declare_partials('R_max', ['R0', 'A'], method="exact")

    def compute_partials(self, inputs, J):
        A = inputs['A']
        R0 = inputs['R0']
        J["R_min", "R0"] = 1 - 1 / A
        J["R_min", "A"] = R0 / A**2
        J["R_max", "R0"] = 1 + 1 / A
        J["R_max", "A"] = -R0 / A**2


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = PlasmaGeometry()

    prob.model.kappa_multiplier = 0.95
    prob.model.κ_ε_scaling_constants = [1.9, 1.9, 1.4]

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('A', 1.6)

    prob.run_driver()
    prob.model.list_outputs()
