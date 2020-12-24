from scipy.constants import mu_0, pi, electron_mass
from scipy.special import hyp2f1
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


class SlowingThermalizationTime(om.ExplicitComponent):
    r"""Time to slow down to zero velocity

    Inputs
    ------
    W/Wc : float
        Initial beam energy to critical energy
    ts : float
        s, Ion-electron slowing down time (Spitzer, 1962)

    Outputs
    -------
    τth : float
        s, Thermalization time
           (time to slow to zero average velocity, in this model)
    """
    def setup(self):
        self.add_input("W/Wc", desc="Initial beam energy / critical energy")
        self.add_input("ts", units="s", desc="Ion-electron slowing time")
        self.add_output("τth")

    def compute(self, inputs, outputs):
        w_rat = inputs["W/Wc"]
        ts = inputs["ts"]
        τth = ts / 3 * np.log(1 + w_rat**(3 / 2))
        outputs["τth"] = τth

    def setup_partials(self):
        self.declare_partials("τth", ["W/Wc", "ts"])

    def compute_partials(self, inputs, J):
        w_rat = inputs["W/Wc"]
        ts = inputs["ts"]
        J["τth", "ts"] = 1 / 3 * np.log(1 + w_rat**(3 / 2))
        J["τth", "W/Wc"] = ts * w_rat**(1 / 2) / (2 * (1 + w_rat**(3 / 2)))

class FastParticleHeatingFractions(om.ExplicitComponent):
    r"""Fraction of fast particle energy to ions vs electrons

    Inputs
    ------
    W/Wc : float
        Initial beam energy to critical energy

    Outputs
    -------
    f_i : float
        Fraction of energy to ions
    f_i : float
        Fraction of energy to electrons
    """
    def setup(self):
        self.add_input("W/Wc", desc="Initial beam energy / critical energy")
        self.add_output("f_i", val=0.5)
        self.add_output("f_e", val=0.5)

    def ionfrac(self, w_rat):
        return hyp2f1(2/3, 1, 5/3, -w_rat**(3/2))

    def compute(self, inputs, outputs):
        w_rat = inputs["W/Wc"]
        f_i = self.ionfrac(w_rat)
        f_e = 1 - f_i
        outputs["f_i"] = f_i
        outputs["f_e"] = f_e

    def setup_partials(self):
        self.declare_partials("f_[ie]", "W/Wc")

    def compute_partials(self, inputs, J):
        w_rat = inputs["W/Wc"]
        dfidw = (1 / (1 + w_rat**(3 / 2)) - self.ionfrac(w_rat)) / w_rat
        J["f_i", "W/Wc"] = dfidw
        J["f_e", "W/Wc"] = -dfidw


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
        numer = Te * u * (α + 2 * At * dαdAt)
        denom = (3 * me * (At * u * β / me)**(2 / 3) * α**(1 / 3))
        J["W_crit", "At"] = numer / denom

        dβdne = (4 / (3 * pi**(1 / 2)))
        numer = -2 * mass_scale * Te * α**(2 / 3) * dβdne
        J["W_crit", "ne"] = numer / (3 * β**(5 / 3))

        dαdni = (1 + At / Ai) * zi**2
        numer = (2 / 3) * mass_scale * Te * dαdni
        denom = (β**(2 / 3) * α**(1 / 3))
        J["W_crit", "ni"] = numer / denom

        dαdzi = 2 * (1 + At / Ai) * ni * zi
        numer = (2 / 3) * mass_scale * Te * dαdzi
        J["W_crit", "Zi"] = numer / denom

        dαdAi = -(At / Ai**2) * ni * zi**2
        numer = (2 / 3) * mass_scale * Te * dαdAi
        J["W_crit", "Ai"] = numer / denom


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
