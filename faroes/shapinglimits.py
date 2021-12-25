# note: this class is not yet integrated into any larger models
import openmdao.api as om


class ZohmMaximumKappaScaling(om.ExplicitComponent):
    r"""Maximum controllable elongation

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
    Used in PROCESS [3]_, which cites Hartmann [2]_,
    but is actually from Zohm [1]_.  In [2]_ a leading value of 1.46
    rather than 1.5 is written, in formula (2.167).

    Zohm writes "The maximum controllable elongation and triangularity for a
    given PF coil will depend on [shape and aspect ratio] and a simple relation
    :math:`\kappa_{X,max} = 1.5 + 0.5/(A-1)` is proposed. Future studies will
    provide more complete fits, also taking into account the :math:`l_i` and
    :math:`\delta`-dependence."

    References
    ----------
    ..[1] Zohm, H.; Angioni, C.; Fable, E.; Federici, G.; Gantenbein, G.;
      Hartmann, T.; Lackner, K.; Poli, E.; Porte, L.; Sauter, O.;
      Tardini, G.; Ward, D.; Wischmeier, M.
      On the Physics Guidelines for a Tokamak DEMO.
      Nuclear Fusion 2013, 53 (7).
      https://doi.org/10.1088/0029-5515/53/7/073019.

    ..[2] Hartmann, T.
      Development of a Modular Systems Code to Analyse the
      Implications of Physics Assumptions on the Design of a
      Demonstration Fusion Power Plant.
      PhD thesis, Technischen Universitat Munchen, Munich, Germany, 2013.

    ..[3] Kovari, M.; Kemp, R.; Lux, H.; Knight, P.; Morris, J.; Ward, D. J.
      "PROCESS": A Systems Code for Fusion Power Plants—Part 1: Physics.
      Fusion Engineering and Design 2014, 89 (12), 3054–3069.
      https://doi.org/10.1016/j.fusengdes.2014.09.018.
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
