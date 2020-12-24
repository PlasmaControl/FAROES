from scipy.constants import mu_0, pi, electron_mass
import numpy as np
import openmdao.api as om
from openmdao.utils.units import unit_conversion

from faroes.units import add_local_units


class AlfvenSpeed(om.ExplicitComponent):
    r"""Alfvén speed

    .. math::

       V_A = \left|B\right| / \sqrt{\mu_0 \rho}

    Inputs
    ------
    |B| : float
        T, Total magnetic field strength
    ρ : float
        kg/m**3, Plasma mass density

    Outputs
    -------
    V_A : float
        m/s, Alfvén speed

    References
    ----------
    https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.parameters.Alfven_speed.html
    """
    def setup(self):
        self.add_input("|B|", units="T", desc="Magnetic field strength")
        self.add_input("ρ", units="kg/m**3", desc="Plasma mass density")
        self.add_output("V_A",
                        units="m/s",
                        ref=1e4,
                        lower=0,
                        desc="Alfvén speed")

    def compute(self, inputs, outputs):
        B = inputs["|B|"]
        ρ = inputs["ρ"]
        V_A = B / (ρ * mu_0)**(1 / 2)
        outputs["V_A"] = V_A

    def setup_partials(self):
        self.declare_partials("V_A", ["ρ", "|B|"])

    def compute_partials(self, inputs, J):
        B = inputs["|B|"]
        ρ = inputs["ρ"]
        J["V_A", "ρ"] = -B * mu_0 / (2 * (mu_0 * ρ)**(3 / 2))
        J["V_A", "|B|"] = 1 / (mu_0 * ρ)**(1 / 2)


class CriticalSlowingEnergy(om.ExplicitComponent):
    r"""Critical energy for fast particles slowing down

    where energy is transferred equally to ions and electrons

    Inputs
    ------
    At : float
        u, test particle mass
    ni : array
        m**-3, ion densities
    mi : array
        u, ion masses
    Zi : array
        units of fundamental charge
    ne : float
        m**-3, electron density
    Te : float
        eV, electron temperature

    Outputs
    -------
    W_crit : float
        eV, Critical energy
    """
    def setup(self):
        add_local_units()
        self.add_input('At', units="u", desc="Test particle atomic mass")
        self.add_input('ne', units='n20', desc="Electron density")
        self.add_input('Te', units='keV', desc="Electron temperature")
        self.add_input('ni',
                       units='n20',
                       shape_by_conn=True,
                       desc="Ion field particles densities")
        self.add_input('Ai',
                       units='u',
                       shape_by_conn=True,
                       copy_shape='ni',
                       desc="Ion field particle atomic masses")
        self.add_input('Zi',
                       shape_by_conn=True,
                       copy_shape='ni',
                       desc="Ion field particle charges")

        self.add_output('W_crit',
                        units='keV',
                        desc='Critical energy for slowing ions')

    def compute(self, inputs, outputs):
        u = unit_conversion('u', 'kg')[0]

        ni = inputs['ni']
        Ai = inputs['Ai']
        me = electron_mass
        zi = inputs['Zi']
        At = inputs['At']
        ne = inputs['ne']
        Te = inputs['Te']
        α = np.sum(ni * zi**2 * (1 + At / Ai))
        β = 4 / (3 * pi**(1 / 2)) * ne
        w_crit = Te * ((At * u / me)**(1 / 3)) * (α / β)**(2 / 3)
        outputs["W_crit"] = w_crit

    def setup_partials(self):
        self.declare_partials("W_crit", ["At", "ne", "Te", "ni", "Ai", "Zi"])

    def compute_partials(self, inputs, J):
        u = unit_conversion('u', 'kg')[0]

        ni = inputs['ni']
        Ai = inputs['Ai']
        me = electron_mass
        zi = inputs['Zi']
        At = inputs['At']
        ne = inputs['ne']
        Te = inputs['Te']
        α = np.sum(ni * zi**2 * (1 + At / Ai))
        β = 4 / (3 * pi**(1 / 2)) * ne
        mass_scale = (At * u / me)**(1 / 3)
        J["W_crit", "Te"] = mass_scale * (α / β)**(2 / 3)

        dαdAt = np.sum(ni * zi**2 / Ai)
        J["W_crit",
          "At"] = Te * u * (α + 2 * At * dαdAt) / (3 * me *
                                                   (At * u * β / me)**(2 / 3) *
                                                   α**(1 / 3))
        dβdne = (4 / (3 * pi**(1 / 2)))
        J["W_crit",
          "ne"] = -2 * mass_scale * Te * α**(2 / 3) * dβdne / (3 * β**(5 / 3))

        dαdni = (1 + At / Ai) * zi**2
        J["W_crit",
          "ni"] = (2 / 3) * mass_scale * Te * dαdni / (β**(2 / 3) * α**(1 / 3))
        dαdzi = 2 * (1 + At / Ai) * ni * zi
        J["W_crit",
          "Zi"] = (2 / 3) * mass_scale * Te * dαdzi / (β**(2 / 3) * α**(1 / 3))
        dαdAi = -(At / Ai**2) * ni * zi**2
        J["W_crit",
          "Ai"] = (2 / 3) * mass_scale * Te * dαdAi / (β**(2 / 3) * α**(1 / 3))


if __name__ == "__main__":
    from openmdao.utils.assert_utils import assert_check_partials
    from scipy.constants import m_p
    prob = om.Problem()
    add_local_units()

    prob.model.add_subsystem('ivc',
                             om.IndepVarComp('ni', val=np.ones(2),
                                             units='n20'),
                             promotes_outputs=["*"])
    prob.model.add_subsystem('cse',
                             CriticalSlowingEnergy(),
                             promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)

    prob.set_val("At", 2 * m_p, units='kg')
    prob.set_val("ne", 1.0e20, units='m**-3')
    prob.set_val("Te", 1.0, units='keV')
    prob.set_val("ni", np.array([0.5e20, 0.5e20]), units='m**-3')
    prob.set_val("Ai", [2, 3], units='u')
    prob.set_val("Zi", [1, 1])

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)

    prob.run_driver()
    all_outputs = prob.model.list_outputs(values=True, print_arrays=True)
