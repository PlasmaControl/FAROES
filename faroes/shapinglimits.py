import openmdao.api as om


class MenardKappaScaling(om.ExplicitComponent):
    r"""

    Elongation :math:`κ` is determined by a "marginal scaling"
    from :footcite:t:`menard_aspect_2004`,

    .. math:: κ = 0.95 (1.9 + 1.9 / (A^{1.4})).

    A factor ":math:`κ` area fraction" from the configuration file
    multiplies the above "marginal :math:`κ`"
    to generate :math:`κ_a`. The latter is used for computations
    of the plasma cross-sectional area and volume.

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

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
        self.options.declare('config', default=None, recordable=False)

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
        aspect_ratio: array_like

        Returns
        -------
        array_like
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


class ZohmMaximumKappaScaling(om.ExplicitComponent):
    r"""Maximum controllable elongation

    .. math:: κ = 1.5 + 0.5 / (A - 1)

    Inputs
    ------
    A : float
        Aspect ratio

    Outputs
    -------
    κ : float
        Maximum elongation

    Notes
    -----
    Used in PROCESS :footcite:p:`kovari_process_2014`, which cites
    :footcite:t:`hartmann_development_2013`, but is actually from
    :footcite:t:`zohm_physics_2013`. In Hartmann formula (2.167) a
    leading value of 1.46 rather than 1.5 is written.

    Zohm writes "The maximum controllable elongation and triangularity for a
    given PF coil will depend on [shape and aspect ratio] and a simple relation
    :math:`κ_{X,max} = 1.5 + 0.5/(A-1)` is proposed. Future studies will
    provide more complete fits, also taking into account the :math:`l_i` and
    :math:`δ`-dependence."
    """
    def setup(self):
        self.add_input("A", desc="Aspect Ratio")
        self.add_output("κ",
                        lower=0,
                        ref=2,
                        desc="Maximum controllable elongation")

    def compute(self, inputs, outputs):
        A = inputs["A"]
        outputs["κ"] = 1.5 + 0.5 / (A - 1)

    def setup_partials(self):
        self.declare_partials("κ", "A")

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        J["κ", "A"] = -0.5 / (A - 1)**2


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = ZohmMaximumKappaScaling()

    prob.setup()

    prob.set_val('A', 2.6)

    prob.run_driver()
    prob.model.list_inputs()
    prob.model.list_outputs()
