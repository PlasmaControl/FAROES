from plasmapy.particles import nuclear_reaction_energy
import astropy.units as u
import faroes.units  # noqa: F401

import openmdao.api as om
from openmdao.api import unit_conversion

from scipy.constants import mega


class SimpleRateCoeff(om.ExplicitComponent):
    r"""Simple fusion rate coefficient

    .. math::

       \sigma v = 1.1 \times 10^{-24} T^2

    Inputs
    ------
    T : float
        keV, Ion temperature

    Outputs
    -------
    <σv> : float
        m**3/s, Fusion reaction rate coefficient

    """
    def setup(self):
        self.coeff = 1.1e-24
        sigma_v_ref = self.coeff * 10**2
        self.add_input("T", units="keV", desc="Ion temperature")
        self.add_output("<σv>",
                        lower=0,
                        units="m**3/s",
                        ref=sigma_v_ref,
                        desc="Rate coefficient")

    def compute(self, inputs, outputs):
        T = inputs["T"]
        c = self.coeff
        outputs["<σv>"] = c * T**2

    def setup_partials(self):
        self.declare_partials("<σv>", "T")

    def compute_partials(self, inputs, J):
        T = inputs["T"]
        c = self.coeff
        J["<σv>", "T"] = 2 * T * c


class VolumetricThermalFusionRate(om.ExplicitComponent):
    r"""

    Inputs
    ------
    n_D : float
        n20, Deuterium density
    n_T : float
        n20, Tritium density
    <σv> : float
        um**3/s, Rate coefficient

    Outputs
    -------
    P_fus/V : float
        MW/m**3, Volumetric fusion energy production rate
    P_n/V : float
        MW/m**3, Volumetric neutron energy production rate
    P_α/V : float
        MW/m**3, Volumetric alpha energy production rate
    """
    def setup(self):
        REACTION_ENERGY = nuclear_reaction_energy(reactants=['D', 'T'],
                                                  products=['alpha', 'n'])
        self.ENERGY_J = REACTION_ENERGY.to(u.J).value
        self.α_fraction = 1 / 5
        self.n_fraction = 4 / 5

        n_units = "n20"
        σv_units = "um**3/s"

        self.add_input("n_D", units=n_units, desc="Deuterium density")
        self.add_input("n_T", units=n_units, desc="Tritium density")
        self.add_input("<σv>", units=σv_units, desc="Rate coefficient")

        self.add_output("P_fus/V",
                        lower=0,
                        units="MW/m**3",
                        desc="Volumetric fusion rate")
        self.add_output("P_n/V",
                        lower=0,
                        units="MW/m**3",
                        desc="Volumetric neutron energy production")
        self.add_output("P_α/V",
                        lower=0,
                        units="MW/m**3",
                        desc="Volumetric α energy production")

        self.n_conv = unit_conversion(n_units, 'm**-3')[0]
        self.sigma_conv = unit_conversion(σv_units, 'm**3/s')[0]

    def compute(self, inputs, outputs):
        n_D = self.n_conv * inputs["n_D"]
        n_T = self.n_conv * inputs["n_T"]
        σv_avg = self.sigma_conv * inputs["<σv>"]

        # W / m**3
        rate = n_D * n_T * σv_avg * self.ENERGY_J

        outputs["P_fus/V"] = rate / mega
        outputs["P_n/V"] = self.n_fraction * rate / mega
        outputs["P_α/V"] = self.α_fraction * rate / mega

    def setup_partials(self):
        self.declare_partials("P_fus/V", ["n_D", "n_T", "<σv>"])
        self.declare_partials("P_n/V", ["n_D", "n_T", "<σv>"])
        self.declare_partials("P_α/V", ["n_D", "n_T", "<σv>"])

    def compute_partials(self, inputs, J):
        n_D = self.n_conv * inputs["n_D"]
        n_T = self.n_conv * inputs["n_T"]
        σv_avg = self.sigma_conv * inputs["<σv>"]

        J["P_fus/V", "n_D"] = self.n_conv * n_T * σv_avg * self.ENERGY_J / mega
        J["P_fus/V", "n_T"] = self.n_conv * n_D * σv_avg * self.ENERGY_J / mega
        J["P_fus/V",
          "<σv>"] = self.sigma_conv * n_D * n_T * self.ENERGY_J / mega
        J["P_α/V", "n_D"] = self.α_fraction * J["P_fus/V", "n_D"]
        J["P_α/V", "n_T"] = self.α_fraction * J["P_fus/V", "n_T"]
        J["P_α/V", "<σv>"] = self.α_fraction * J["P_fus/V", "<σv>"]
        J["P_n/V", "n_D"] = self.n_fraction * J["P_fus/V", "n_D"]
        J["P_n/V", "n_T"] = self.n_fraction * J["P_fus/V", "n_T"]
        J["P_n/V", "<σv>"] = self.n_fraction * J["P_fus/V", "<σv>"]


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = VolumetricThermalFusionRate()

    prob.setup(force_alloc_complex=True)

    prob.set_val("<σv>", 1.1e-24, units="m**3/s")
    prob.set_val("n_D", 0.5, units="n20")
    prob.set_val("n_T", 0.5, units="n20")

    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
