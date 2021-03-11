import openmdao.api as om
import numpy as np
from openmdao.utils.assert_utils import assert_check_partials

from faroes.plasma_beta import ThermalBetaPoloidal


class BootstrapMultiplier(om.ExplicitComponent):
    r"""

    The 'bootstrap multiplier' is given by

    .. math::

        \max(1.2 - (q^* / q_{min} / 5), 0.6)

    Notes
    -----
    In terms of physics, I don't know where this expression comes from.

    We'd like to approximate the `max` function in a way such that the first
    derivatives are continuous. One way is to use a LogSumExp function,

    .. math::

       \max(a, b) \approx \log({base}^a + {base}^b) / \log({base})

    but this turns out to be bad numerically as `base` needs to be a large
    number, like 10^5. Instead we use

    .. math::

       \frac{1}{s f q_{min}} \log(1 + \exp((q_{min} f y_1 - q^*) s)) + y_1

    where :math:`f = 5`, and :math:`y_1 = 0.6`, and :math:`s` is a sharpness
    parameter.

    Inputs
    ------
    q_star : float
        Normalized q
    q_min : float
        Minimum q

    Outputs
    -------
    bs_mult : float
        Bootstrap multiplier
    """

    def setup(self):
        self.f = 5
        self.y1 = 0.6
        self.s = 2.0  # sharpness factor
        self.add_input("q_star", val=3.5)
        self.add_input("q_min", val=2.2)
        self.add_output("bs_mult")

    def compute(self, inputs, outputs):
        f = self.f
        y1 = self.y1
        s = self.s
        qs = inputs["q_star"]
        qm = inputs["q_min"]

        if qs > 40:
            raise om.AnalysisError(f"q_star = {qs} > 40")
        if qs < 0:
            raise om.AnalysisError(f"q_star = {qs} < 0")

        e1 = qm * y1 * f - qs
        bs_mult = 1 / (s * qm * f) * np.log(1 + np.exp(s * e1)) + y1
        outputs["bs_mult"] = bs_mult

    def setup_partials(self):
        self.declare_partials("bs_mult", ["q_star", "q_min"])

    def compute_partials(self, inputs, J):
        f = self.f
        y1 = self.y1
        s = self.s
        qs = inputs["q_star"]
        qm = inputs["q_min"]
        J["bs_mult", "q_star"] = -1 / \
            (f * qm + f * qm * np.exp(s * (qs - f * qm * y1)))

        term1 = np.exp(s * f * qm * y1) * qm * y1 / \
            (np.exp(qs * s) + np.exp(f * qm * y1 * s))
        term2 = -1 / (s*f) * np.log(1 + np.exp(-qs * s + s*f*qm*y1))
        J["bs_mult", "q_min"] = (term1 + term2)/qm**2


class BootstrapFraction(om.ExplicitComponent):
    r"""

    .. math::

        0.9 bs_\mathrm{mult} * \beta_{p, th} * \sqrt{\epsilon}

    Inputs
    ------
    ε : float
        Inverse aspect ratio
    bs_mult : float
        Boostrap multiplier based on q*/q_min
    βp_th : float
        Thermal poloidal beta

    Outputs
    -------
    f_BS : float
        Bootstrap fraction

    Notes
    -----
    I don't know where this formula came from.
    """

    def setup(self):
        self.add_input("bs_mult", desc="Bootstrap multiplier")
        self.add_input("ε")
        self.add_input("βp_th")
        self.add_output("f_BS")
        self.c = 0.9  # boostrap fraction multiplier

    def compute(self, inputs, outputs):
        ε = inputs["ε"]
        bs_mult = inputs["bs_mult"]
        βp_th = inputs["βp_th"]
        c = self.c
        outputs["f_BS"] = c * np.sqrt(ε) * bs_mult * βp_th

    def setup_partials(self):
        self.declare_partials("f_BS", ["ε", "bs_mult", "βp_th"])

    def compute_partials(self, inputs, J):
        ε = inputs["ε"]
        bs_mult = inputs["bs_mult"]
        βp_th = inputs["βp_th"]
        c = self.c
        J["f_BS", "ε"] = c * βp_th * bs_mult * (1 / 2) * ε**(-1 / 2)
        J["f_BS", "βp_th"] = c * bs_mult * ε**(1 / 2)
        J["f_BS", "bs_mult"] = c * βp_th * ε**(1 / 2)


class BootstrapCurrent(om.Group):
    r"""
    Inputs
    ------
    ε : float
        Inverse aspect ratio of the plasma
    Ip : float
        MA, plasma current
    βp : float
        Total poloidal beta
    thermal pressure fraction: float
        Fraction of pressure which is thermal
    q_star : float
        Normalized q
    q_min : float
        Minimum q

    Outputs
    -------
    f_BS : float
        Bootstrap current fraction
    I_BS : float
        MA, Boostrap current
    """

    def setup(self):
        self.add_subsystem("beta_pth",
                           ThermalBetaPoloidal(),
                           promotes_inputs=["βp", "thermal pressure fraction"],
                           promotes_outputs=["βp_th"])
        self.add_subsystem("bsm",
                           BootstrapMultiplier(),
                           promotes_inputs=["q_star", "q_min"])
        self.add_subsystem("bsf",
                           BootstrapFraction(),
                           promotes_inputs=["ε", "βp_th"],
                           promotes_outputs=["f_BS"])
        self.connect("bsm.bs_mult", ["bsf.bs_mult"])
        self.add_subsystem("ip",
                           om.ExecComp("I_BS = f_BS * Ip",
                                       Ip={"units": "MA"},
                                       I_BS={"units": "MA"}),
                           promotes=["*"])


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = BootstrapCurrent()
    prob.setup(force_alloc_complex=True)

    prob.set_val('ε', 0.36)
    prob.set_val('βp_th', 1.0)
    prob.set_val('thermal pressure fraction', 0.9)
    prob.set_val('q_min', 2.0)
    prob.set_val('q_star', 3.0)
    prob.set_val('Ip', 14.0, units="MA")

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)
    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
