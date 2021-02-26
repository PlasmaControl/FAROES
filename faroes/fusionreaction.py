from plasmapy.particles import nuclear_reaction_energy, Particle
from openmdao.utils.assert_utils import assert_check_partials
import astropy.units as u
import faroes.units  # noqa: F401

import openmdao.api as om
from openmdao.api import unit_conversion

from scipy.constants import mega, kilo, Avogadro, femto

_REACTION_ENERGY = nuclear_reaction_energy(reactants=['D', 'T'],
                                           products=['alpha', 'n'])
_ENERGY_J = _REACTION_ENERGY.to(u.J).value
_DT_fusion_data = {
    "REACTION_ENERGY": _REACTION_ENERGY,
    "ENERGY_J": _ENERGY_J,
    "α_fraction": 1 / 5,
    "n_fraction": 4 / 5,
}


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
    rate_fus/V: float
        1/m**3/s, Volumetric thermal fusion rate
    P_fus/V : float
        MW/m**3, Volumetric fusion energy production rate
    P_n/V : float
        MW/m**3, Volumetric neutron energy production rate
    P_α/V : float
        MW/m**3, Volumetric alpha energy production rate
    """
    def setup(self):
        data = _DT_fusion_data
        self.ENERGY_J = data["ENERGY_J"]
        self.α_fraction = data["α_fraction"]
        self.n_fraction = data["n_fraction"]

        n_units = "n20"
        σv_units = "um**3/s"

        self.add_input("n_D", units=n_units, desc="Deuterium density")
        self.add_input("n_T", units=n_units, desc="Tritium density")
        self.add_input("<σv>", units=σv_units, desc="Rate coefficient")

        self.add_output("rate_fus/V",
                        lower=0,
                        units="1/m**3/fs",
                        desc="Volumetric fusion rate")
        self.add_output("P_fus/V",
                        lower=0,
                        units="MW/m**3", ref=3,
                        desc="Volumetric fusion energy production")
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

        # / s
        rate = n_D * n_T * σv_avg
        # W / m**3
        energy_rate = rate * self.ENERGY_J
        outputs["rate_fus/V"] = rate * femto
        outputs["P_fus/V"] = energy_rate / mega
        outputs["P_n/V"] = self.n_fraction * energy_rate / mega
        outputs["P_α/V"] = self.α_fraction * energy_rate / mega

    def setup_partials(self):
        self.declare_partials("rate_fus/V", ["n_D", "n_T", "<σv>"])
        self.declare_partials("P_fus/V", ["n_D", "n_T", "<σv>"])
        self.declare_partials("P_n/V", ["n_D", "n_T", "<σv>"])
        self.declare_partials("P_α/V", ["n_D", "n_T", "<σv>"])

    def compute_partials(self, inputs, J):
        n_D = self.n_conv * inputs["n_D"]
        n_T = self.n_conv * inputs["n_T"]
        σv_avg = self.sigma_conv * inputs["<σv>"]

        J["rate_fus/V", "n_D"] = self.n_conv * n_T * σv_avg * femto
        J["rate_fus/V", "n_T"] = self.n_conv * n_D * σv_avg * femto
        J["rate_fus/V", "<σv>"] = self.sigma_conv * n_D * n_T * femto
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


class NBIBeamTargetFusion(om.ExplicitComponent):
    r"""Simple calculation of beam-target DT fusion

    Assumes deuterium beams impinging on a 50% D-T plasma

    .. math::

        \mathrm{Rate} = 80 * 1.1e14 P_\mathrm{NBI} \left<T_e\right>^{3/2}

    Inputs
    ------
    P_NBI : float
        MW, Neutral beam injected power
    <T_e> : float
        keV, Averaged plasma temperature

    Outputs
    -------
    rate_fus : float
        1/fs, Fusion reaction rate
    P_fus : float
        MW, Total fusion power
    #P_α : float
    #    MW, Fusion alpha power
    #P_n : float
    #    MW, Fusion neutron power

    Notes
    -----
    From Menard spreadsheet cells T318 and T319. There is a reference
    to [1], though I could not find the formula anywhere in the paper.

    The rightmost part of the formula assumes D-D fusion rates only,
    while the factor 80 represents the rate increase from D-D to 50/50 D/T.

    References
    ----------
    [1] Strachan, J. D.; Colestock, P. L.; Davis, S. L.; Eames, D.;
    Efthimion, P. C.; Eubank, H. P.; Goldston, R. J.; Grisham, L. R.;
    Hawryluk, R. J.; Hosea, J. C.; Hovey, J.; Jassby, D. L.;
    Johnson, D. W.; Mirin, A. A.; Schilling, G.; Stooksberry, R.;
    Stewart, L. D.; Towner, H. H.
    Fusion Neutron Production during Deuterium Neutral-Beam Injection
    into the PLT Tokamak. Nuclear Fusion 1981, 21 (1), 67–81.
    https://doi.org/10.1088/0029-5515/21/1/006.

    """
    def setup(self):
        data = _DT_fusion_data
        self.ENERGY_J = data["ENERGY_J"]
        self.α_fraction = data["α_fraction"]
        self.n_fraction = data["n_fraction"]

        self.constant = 1.1e14
        self.DT_mult = 80
        self.add_input("P_NBI", units="MW", desc="Neutral beam injected power")
        self.add_input("<T_e>", units="keV")

        self.mmol = Avogadro / kilo

        R_fus_ref = 1e4
        self.add_output("rate_fus", units="1/fs", ref=R_fus_ref, lower=0)
        P_fus_ref = 10
        self.add_output("P_fus", units="MW", ref=P_fus_ref, lower=0)
        # self.add_output("P_α", units="MW", ref=P_fus_ref, lower=0)
        # self.add_output("P_n", units="MW", ref=P_fus_ref, lower=0)

    def compute(self, inputs, outputs):
        c = self.constant * self.DT_mult
        P_NBI = inputs["P_NBI"]
        Te = inputs["<T_e>"]

        R = c * P_NBI * Te**(3 / 2)
        outputs["rate_fus"] = R * femto

        energy_MJ = self.ENERGY_J / mega
        P_fus = R * energy_MJ
        outputs["P_fus"] = P_fus
        # P_α = P_fus * self.α_fraction
        # P_n = P_fus * self.n_fraction
        # outputs["P_α"] = P_α
        # outputs["P_n"] = P_n

    def setup_partials(self):
        self.declare_partials(
            [
                "rate_fus",
                "P_fus",
                # "P_α", "P_n",
            ],
            ["P_NBI", "<T_e>"])

    def compute_partials(self, inputs, J):
        c = self.constant * self.DT_mult
        P_NBI = inputs["P_NBI"]
        Te = inputs["<T_e>"]
        energy_MJ = self.ENERGY_J / mega
        J["rate_fus", "P_NBI"] = c * Te**(3 / 2) * femto
        J["rate_fus", "<T_e>"] = (3 / 2) * c * P_NBI * Te**(1 / 2) * femto
        J["P_fus", "P_NBI"] = c * Te**(3 / 2) * energy_MJ
        J["P_fus", "<T_e>"] = (3 / 2) * c * P_NBI * Te**(1 / 2) * energy_MJ
        # J["P_α", "P_NBI"] = self.α_fraction * J["P_fus", "P_NBI"]
        # J["P_α", "<T_e>"] = self.α_fraction * J["P_fus", "<T_e>"]
        # J["P_n", "P_NBI"] = self.n_fraction * J["P_fus", "P_NBI"]
        # J["P_n", "<T_e>"] = self.n_fraction * J["P_fus", "<T_e>"]


class TotalDTFusionRate(om.ExplicitComponent):
    r"""From thermal and beam-target components

    Inputs
    ------
    rate_th : float
        1 / fs, Thermal fusion rate
    rate_NBI : float
        1 / fs, NBI beam-target fusion rate

    P_fus_th : float
        MW, Thermal fusion power
    P_fus_NBI : float
        MW, NBI beam-target fusion power

    Outputs
    -------
    rate_fus : float
        1 / fs
    P_fus : float
        MW, Total fusion power
    P_α : float
        MW, Fusion alpha power
    P_n : float
        MW, Fusion neutron power

    Notes
    -----
    The rates and powers are trivially linked, but they're already calculated
    elsewhere so I don't want to update everything.
    """
    def setup(self):
        data = _DT_fusion_data
        self.α_fraction = data["α_fraction"]
        self.n_fraction = data["n_fraction"]

        self.add_input("P_fus_th", val=0, units="MW")
        self.add_input("P_fus_NBI", val=0, units="MW")
        self.add_input("rate_th", val=0, units="1/fs")
        self.add_input("rate_NBI", val=0, units="1/fs")

        rate_fus_ref=1e5
        self.add_output("rate_fus", lower=0, ref=rate_fus_ref, units="1/fs")
        P_fus_ref = 100
        self.add_output("P_fus", lower=0, ref=P_fus_ref, units="MW")
        self.add_output("P_α", lower=0, ref=P_fus_ref, units="MW")
        self.add_output("P_n", lower=0, ref=P_fus_ref, units="MW")

    def compute(self, inputs, outputs):
        P_fus_th = inputs["P_fus_th"]
        P_fus_NBI = inputs["P_fus_NBI"]
        P_fus = P_fus_th + P_fus_NBI
        outputs["rate_fus"] = inputs["rate_th"] + inputs["rate_NBI"]
        outputs["P_fus"] = P_fus
        outputs["P_α"] = P_fus * self.α_fraction
        outputs["P_n"] = P_fus * self.n_fraction

    def setup_partials(self):
        self.declare_partials("rate_fus", ["rate_th", "rate_NBI"], val=1.0)
        self.declare_partials("P_fus", ["P_fus_th", "P_fus_NBI"], val=1.0)
        self.declare_partials("P_α", ["P_fus_th", "P_fus_NBI"],
                              val=self.α_fraction)
        self.declare_partials("P_n", ["P_fus_th", "P_fus_NBI"],
                              val=self.n_fraction)


class SimpleFusionAlphaSource(om.ExplicitComponent):
    r"""Source for alpha particle properties

    Outputs
    -------
    E : float
        keV, energy per particle
    A : float
        u, mass of particles
    Z : int
        e, Charge of beam particles
    """
    def setup(self):
        self.add_input("rate", val=0, units="1/s")
        # number of particles in a millimole
        self.mmol = Avogadro / kilo

        data = _DT_fusion_data
        E = data["REACTION_ENERGY"].to(u.keV).value
        self.α_fraction = data["α_fraction"]
        E_α = E * self.α_fraction

        alpha = Particle("alpha")
        m_α = alpha.nuclide_mass.to(u.kg).value
        A_α = alpha.nuclide_mass.to(u.u).value
        Z_α = alpha.integer_charge

        E_J = data["REACTION_ENERGY"].to(u.J).value * self.α_fraction
        v_α = (2 * E_J / m_α)**(1 / 2)

        self.add_output("S", val=0, units="1/s", ref=1e20)
        self.add_output("E", val=E_α, units="keV", ref=3.5e3)
        self.add_output("A", val=A_α, units="u")
        self.add_output("Z", val=Z_α)
        self.add_output("v", val=v_α, units="m/s", ref=10e6)


if __name__ == "__main__":
    prob = om.Problem()

    # prob.model = VolumetricThermalFusionRate()
    prob.model = NBIBeamTargetFusion()

    prob.setup(force_alloc_complex=True)

    # prob.set_val("<σv>", 1.1e-24, units="m**3/s")
    # prob.set_val("n_D", 0.5, units="n20")
    # prob.set_val("n_T", 0.5, units="n20")
    prob.set_val("P_NBI", 50, units="MW")
    prob.set_val("<T_e>", 9.2, units="keV")

    prob.run_driver()
    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)
    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
