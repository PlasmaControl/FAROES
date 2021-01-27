import openmdao.api as om
import numpy as np
from openmdao.utils.assert_utils import assert_check_partials

from faroes.plasma_beta import ThermalBetaPoloidal


class BootstrapMultiplier(om.ExplicitComponent):
    r"""

    .. math::

        \max(1.2 - (q^* / q_{min} / 5), 0.6)

    Notes
    -----
    I don't know where this expression comes from.
    Implemented using a LogSumExp function

    .. math::

       \max(a, b) \approx \log({base}^a + {base}^b) / \log({base})

    Inputs
    ------
    qstar : float
        Normalized q
    qmin : float
        Minimum q

    Outputs
    -------
    bs_mult : float
        Bootstrap multiplier
    """
    def setup(self):
        self.y0 = 1.2
        self.f = 5
        self.y1 = 0.6
        self.b = 10**5  # can be a fairly large arbitrary number
        self.add_input("qstar")
        self.add_input("qmin")
        self.add_output("bs_mult")

    def compute(self, inputs, outputs):
        y0 = self.y0
        f = self.f
        y1 = self.y1
        b = self.b
        qs = inputs["qstar"]
        qm = inputs["qmin"]
        e1 = y0 - (qs / qm) / f
        e2 = y1
        bs_mult = np.log(b**e1 + b**e2) / np.log(b)
        outputs["bs_mult"] = bs_mult

    def setup_partials(self):
        self.declare_partials("bs_mult", ["qstar", "qmin"])

    def compute_partials(self, inputs, J):
        y0 = self.y0
        f = self.f
        y1 = self.y1
        b = self.b
        qs = inputs["qstar"]
        qm = inputs["qmin"]
        J["bs_mult",
          "qstar"] = -1 / (f * qm + f * qm * b**(y1 + qs / (f * qm) - y0))
        J["bs_mult",
          "qmin"] = b**y0 * qs / ((b**y0 + b**(qs /
                                               (f * qm) + y1)) * f * qm**2)


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
                           promotes_inputs=["qstar", "qmin"])
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
    prob.set_val('qmin', 2.0)
    prob.set_val('qstar', 2.0)
    prob.set_val('Ip', 14.0, units="MA")

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)
    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
