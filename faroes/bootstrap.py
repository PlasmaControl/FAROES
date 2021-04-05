import openmdao.api as om
import numpy as np
from openmdao.utils.assert_utils import assert_check_partials

from faroes.plasma_beta import ThermalBetaPoloidal

from faroes.trappedparticles import SauterTrappedParticleFraction
from math import sqrt


class SauterBootstrapProportionality(om.Group):
    r"""Allows adjustment of bootstrap fraction with δ.

    The local bootstrap current is proportional to the trapped particle
    fraction [1]_

    .. math::

        j_B \propto f_t / (1 - f_t)


    This is a correction factor to the existing bootstrap current
    formula, so it is constructed to be unity when δ = 0.

    Thus,

    .. math::

       m = \left(\frac{f_t(\epsilon, \delta)}{1 - f_t(\epsilon, \delta)}\right)
           \left(\frac{f_t(\epsilon, \delta=0)}
           {1 - f_t(\epsilon, \delta=0)}\right)^{-1}.

    Here we assume that ε=0.75 of the LCFS ε and δ is 0.5 of the LCFS δ.
    This assumes that :math:`\delta = \delta_0(1 - \sqrt{1 - 0.75})`, i.e.,
    delta increases nonlinearly toward the LCFS. This scaling is similar to
    that shown in Figure D24 of [2]_.

    Inputs
    ------
    ε: float
       Inverse aspect ratio of the plasma LCFS
    δ: float
       Triangularity of the plasma LCFS

    Outputs
    -------
    m: float
       Multiplier of bootstrap current relative to when δ=0 case.

    References
    ----------
    .. [1] Kikuchi, M.; Takizuka, T.; Furukawa, M.
       Negative Triangularity as a Possible Tokamak Scenario.
       Proceedings of the 12th Asia Pacific Physics Conference (APPC12) 2014.
       https://doi.org/10.7566/JPSCP.1.015014.

    .. [2] Sauter, O. Geometric Formulas for System Codes
       Including the Effect of Negative Triangularity.
       Fusion Engineering and Design 2016, 112, 633–645.
       https://doi.org/10.1016/j.fusengdes.2016.04.033.
    """
    def setup(self):
        self.set_input_defaults("δ", 0)
        ivc = om.IndepVarComp()
        ivc.add_output("δ0", val=0)
        self.add_subsystem("ivc", ivc)

        self.add_subsystem("new_eps",
                           om.ExecComp("eps = 0.75 * oldeps",
                                       oldeps={'value': 0.5},
                                       eps={'value': 0.5}),
                           promotes_inputs=[("oldeps", "ε")],
                           promotes_outputs=[("eps", "ε_new")])

        self.add_subsystem("new_delta",
                           om.ExecComp(
                               "delta = 0.5 * olddelta",
                               olddelta={'value': 0.5},
                               delta={'value': 0.5},
                           ),
                           promotes_inputs=[("olddelta", "δ")],
                           promotes_outputs=[("delta", "δ_new")])

        self.add_subsystem("ft",
                           SauterTrappedParticleFraction(),
                           promotes_inputs=[("ε", "ε_new"), ("δ", "δ_new")],
                           promotes_outputs=[("ftrap", "ft")])
        self.add_subsystem("ft0",
                           SauterTrappedParticleFraction(),
                           promotes_inputs=[("ε", "ε_new")],
                           promotes_outputs=[("ftrap", "ft0")])
        self.connect("ivc.δ0", "ft0.δ")
        self.add_subsystem("mult",
                           om.ExecComp("m = ft/(1-ft) * (1 - ft0)/ft0",
                                       ft={'value': 0.5},
                                       ft0={'value': 0.5}),
                           promotes_inputs=["ft", "ft0"],
                           promotes_outputs=["m"])


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

        if qs > 80:
            raise om.AnalysisError(f"q_star = {qs} > 80")
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
        term2 = -1 / (s * f) * np.log(1 + np.exp(-qs * s + s * f * qm * y1))
        J["bs_mult", "q_min"] = (term1 + term2) / qm**2


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
    δ_mult : float
        Extra multiplier to incorporate triangularity effects.

    Outputs
    -------
    f_BS : float
        Bootstrap fraction

    Notes
    -----
    I don't know where this formula came from. It is similar to a formula in
    Wesson [1] for low-aspect-ratio tokamaks,

    .. math::

       f_{BS} = c \epsilon^{1/2} \beta_p,

    found at the end of Section 4.9 (Bootstrap current).

    References
    ----------
    .. [1] Wesson, J. Tokamaks, 3rd ed.;
       Oxford University Press: New York, 2004.
    """
    def setup(self):
        self.add_input("bs_mult", desc="Bootstrap multiplier")
        self.add_input("δ_mult", desc="Extra multiplier for dependence on δ")
        self.add_input("ε", val=0.5)
        self.add_input("βp_th")
        self.add_output("f_BS")
        self.c = 0.9  # boostrap fraction multiplier

    def compute(self, inputs, outputs):
        ε = inputs["ε"]
        bs_mult = inputs["bs_mult"]
        δ_mult = inputs["δ_mult"]
        βp_th = inputs["βp_th"]
        c = self.c
        outputs["f_BS"] = c * ε**(1 / 2) * bs_mult * βp_th * δ_mult

    def setup_partials(self):
        self.declare_partials("f_BS", ["ε", "bs_mult", "βp_th", "δ_mult"])

    def compute_partials(self, inputs, J):
        ε = inputs["ε"]
        bs_mult = inputs["bs_mult"]
        βp_th = inputs["βp_th"]
        δ_mult = inputs["δ_mult"]
        c = self.c
        J["f_BS", "ε"] = c * βp_th * bs_mult * (1 / 2) / sqrt(ε) * δ_mult
        J["f_BS", "βp_th"] = c * bs_mult * sqrt(ε) * δ_mult
        J["f_BS", "bs_mult"] = c * βp_th * sqrt(ε) * δ_mult
        J["f_BS", "δ_mult"] = c * βp_th * sqrt(ε)


class BootstrapCurrent(om.Group):
    r"""
    Inputs
    ------
    ε : float
        Inverse aspect ratio of the plasma
    δ : float
        Triangularity of the plasma.
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
        self.set_input_defaults("δ", 0)
        self.add_subsystem("triangularity_factor",
                           SauterBootstrapProportionality(),
                           promotes_inputs=["ε", "δ"])
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
        self.connect("triangularity_factor.m", ["bsf.δ_mult"])
        self.add_subsystem("ip",
                           om.ExecComp("I_BS = f_BS * Ip",
                                       Ip={"units": "MA"},
                                       I_BS={"units": "MA"}),
                           promotes=["*"])


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = BootstrapCurrent()
    # prob.model = SauterBootstrapProportionality()
    prob.setup(force_alloc_complex=True)

    prob.set_val('ε', 0.36)
    prob.set_val('δ', 0.5)
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
