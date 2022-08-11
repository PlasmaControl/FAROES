import openmdao.api as om
from openmdao.utils.cs_safe import abs as cs_safe_abs
from faroes.util import SoftCapUnity

import numpy as np


class TrappedParticleFractionUpperEst(om.ExplicitComponent):
    r"""Upper estimate for the trapped particle fraction on a flux surface

    .. math::

       f_{\mathrm{trap},u} = 1 - \left(1 - \epsilon^2\right)^{-1/2}
                                 \left(1 - \frac{3}{2} \epsilon^{1/2} +
                                           \frac{1}{2} \epsilon^{3/2}\right)

    Notes
    -----
    This is derived by :footcite:t:`linliu_upper_1995`, Equation (13),
    for "... the case of concentric, elliptical flux surfaces that
    are adequate to describe low-β, up-down symmetric equilibria"

    This might be enhanced in the future into a 'mid-range'
    trapped-particle-fraction estimate, by programming in the lower estimate
    and the suggested interpolation given in the paper.

    Inputs
    ------
    ε : float
        Inverse aspect ratio of flux surface

    Outputs
    -------
    ftrap_u : float
        Upper estimate of the trapped particle fraction on that surface
    """
    def setup(self):
        self.add_input("ε", desc="Inverse aspect ratio of flux surface")
        self.add_output("ftrap_u",
                        lower=0,
                        val=0.5,
                        desc="Trapped particle fraction")

    def compute(self, inputs, outputs):
        ε = inputs["ε"]
        ftrap_u = 1 - (1 - ε**2)**(-1 / 2) * (1 - (3 / 2) * ε**(1 / 2) +
                                              (1 / 2) * ε**(3 / 2))
        outputs["ftrap_u"] = ftrap_u

    def setup_partials(self):
        self.declare_partials("ftrap_u", ["ε"])

    def compute_partials(self, inputs, J):
        ε = inputs["ε"]
        numer = 3 + ε**(1 / 2) * (3 - ε * (4 + ε**(1 / 2) + ε))
        denom = 4 * (1 + ε**(1 / 2)) * (1 + ε) * (ε - ε**3)**(1 / 2)
        J["ftrap_u", "ε"] = numer / denom


class SauterTrappedParticleFractionCalc(om.ExplicitComponent):
    r"""To be used with SauterTrappedParticleFraction

    See that group for details.
    """
    def setup(self):
        self.add_input("ε", val=0.5, desc="Flux surface inv. asp. ratio")
        self.add_input("δ", val=0.0, desc="Flux surface triangularity")
        # note: the upper limit should never actually be hit, here
        self.add_output("ftrap",
                        lower=0,
                        upper=1.01,
                        val=0.5,
                        desc="Trapped particle fraction")
        self.c0 = 0.67
        self.c1 = 1.4

    def compute(self, inputs, outputs):
        δ = inputs["δ"]
        ε = inputs["ε"]
        c0 = self.c0
        c1 = self.c1
        ε_eff = c0 * (1 - c1 * δ * cs_safe_abs(δ)) * ε
        ft = 1 - (1 - ε_eff) / (1 + 2 * ε_eff**(1 / 2)) * ((1 - ε) /
                                                           (1 + ε))**(1 / 2)
        outputs["ftrap"] = ft

    def setup_partials(self):
        self.declare_partials('ftrap', ['ε', 'δ'])

    def compute_partials(self, inputs, J):
        δ = inputs["δ"]
        ε = inputs["ε"]
        c0 = self.c0
        c1 = self.c1
        dεeff_dε = c0 * (1 - c1 * δ * cs_safe_abs(δ))

        # take care of the abs
        dεeff_dδ = -2 * c0 * c1 * δ * ε
        if δ < 0:
            dεeff_dδ = -dεeff_dδ

        ε_eff = c0 * (1 - c1 * δ * cs_safe_abs(δ)) * ε
        dft_dεeff = (1 + ε_eff**(1 / 2) + ε_eff) * np.sqrt(
            (1 - ε) / (ε_eff + ε * ε_eff)) / (1 + 2 * ε_eff**(1 / 2))**2
        dft_dε = (1 - ε_eff) / ((1 - ε)**(1 / 2) * (1 + ε)**(3 / 2) *
                                (1 + 2 * np.sqrt(ε_eff)))
        J["ftrap", 'ε'] = dft_dε + dft_dεeff * dεeff_dε
        J["ftrap", 'δ'] = dft_dεeff * dεeff_dδ


class SauterTrappedParticleFraction(om.Group):
    r"""Sauter's estimate of trapped particle fraction with triangularity

    :footcite:t:`sauter_geometric_2016` defines

    .. math::

       f_t = 1 -
           \frac{1 - \epsilon_\mathrm{eff}}
           {1 + 2 \sqrt{\epsilon_\mathrm{eff}}}
           \left(\frac{1-\epsilon}{1 +\epsilon}\right)

    where

    .. math::

       \epsilon_\mathrm{eff} =
           0.67 (1 - 1.4 \delta \left|\delta\right|)\epsilon.

    A limiter

    .. math::

        f_t = \min(1, f_t)

    is applied against :math:`f_t` > 1, which can happen in extreme
    scenarios of very small :math:`\epsilon` and negative :math:`\delta`.
    For numerical reasons this is implemented with
    :class:`.utils.SoftCapUnity`.

    Notes
    -----
    I'm not sure if this is for an entire plasma or for a particular
    flux surface.

    Inputs
    ------
    ε : float
       Inverse aspect ratio of a flux surface
    δ : float
       Triangularity of a flux surface

    Outputs
    -------
    ftrap : float
       Estimate of trapped particle fraction
    """
    def setup(self):
        self.add_subsystem("ft",
                           SauterTrappedParticleFractionCalc(),
                           promotes_inputs=["*"])
        self.add_subsystem("soft_limiter",
                           SoftCapUnity(),
                           promotes_outputs=[('y', "ftrap")])
        self.connect("ft.ftrap", "soft_limiter.x")


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = SauterTrappedParticleFraction()

    prob.setup()

    prob.set_val("δ", 0.2)
    prob.set_val("ε", 0.5)

    prob.run_driver()
    all_inputs = prob.model.list_inputs(val=True)
    all_outputs = prob.model.list_outputs(val=True)
