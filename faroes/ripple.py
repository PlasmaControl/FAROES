import openmdao.api as om

from math import log


class SimpleRipple(om.ExplicitComponent):
    r"""Ripple magnitude estimation from Wesson

    Estimates vacuum toroidal field ripple magnitude at the midplane,
    for filamentary currents.  Formula is provided in [1]_. A comparison
    is given in [2]_ to a more sophisticated formula.

    .. math::

       \delta = \left(\frac{R}{r_2}\right)^N + \left(\frac{r_1}{R}\right)^N

    Notes
    -----
    This assumes (infinitely tall?) straight filementary currents.

    Inputs
    ------
    R : float
        m, Radius at which to evaluate ripple
    r1 : float
        m, Inboard leg conductor radius
    r2 : float
        m, Outboard leg conductor radius
    n_coil : int
        Number of TF coils

    Outputs
    -------
    δ : float
        Normalized ripple strength

    References
    ----------
    .. [1] Wesson, J. Tokamaks, 3rd ed.;
       Oxford University Press: New York, 2004.
       Section 4.11: Ripple transport. Page 175.

    .. [2] Kovari, M.; Kemp, R.; Lux, H.; Knight, P.; Morris, J.; Ward, D. J.
       "PROCESS": A Systems Code for Fusion Power Plants—Part 1: Physics.
       Fusion Engineering and Design 2014, 89 (12), 3054–3069.
       https://doi.org/10.1016/j.fusengdes.2014.09.018.
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        self.add_input("R", units="m", desc="Major radius at evaluation point")
        self.add_input("r1",
                       units="m",
                       desc="Inboard TF leg average conductor radius")
        self.add_input("r2",
                       units="m",
                       desc="Outboard TF leg average conductor radius")
        self.add_input("n_coil", 18, desc="Number of coils")

        self.add_output("δ",
                        ref=0.001,
                        lower=0,
                        upper=1,
                        desc="Normalized ripple strength")

    def compute(self, inputs, outputs):
        R = inputs["R"]
        r1 = inputs["r1"]
        r2 = inputs["r2"]
        n = inputs["n_coil"]
        δ = (R / r2)**n + (r1 / R)**n
        outputs["δ"] = δ

    def setup_partials(self):
        self.declare_partials("δ", ["r1", "r2", "R", "n_coil"])

    def compute_partials(self, inputs, J):
        n = inputs['n_coil']
        R = inputs["R"]
        r1 = inputs["r1"]
        r2 = inputs["r2"]

        J["δ", "r1"] = (n / r1) * (r1 / R)**n
        J["δ", "r2"] = -(n / r2) * (R / r2)**n
        J["δ", "R"] = (n / R) * ((R / r2)**n - (r1 / R)**n)
        J["δ",
          "n_coil"] = (r1 / R)**n * log(r1 / R) + (R / r2)**n * log(R / r2)


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = SimpleRipple()

    prob.setup()

    prob.set_val("R", 5)
    prob.set_val("r2", 8.168)
    prob.set_val("r1", 0.26127)
    prob.set_val("n_coil", 18)

    prob.run_driver()
    all_outputs = prob.model.list_outputs(values=True)
