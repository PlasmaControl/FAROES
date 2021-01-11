import faroes.units  # noqa: F401

import openmdao.api as om
from openmdao.utils.units import unit_conversion

from scipy.constants import pi, electron_mass, kilo, mega, eV
from scipy.special import hyp2f1
import numpy as np


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
        return hyp2f1(2 / 3, 1, 5 / 3, -w_rat**(3 / 2))

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


class SlowingTimeOnElectrons(om.ExplicitComponent):
    r"""Slowing time of ions on electrons, from Spitzer

    Inputs
    ------
    At : float
        u, Test particle mass
    Zt : int
        units of fundamental charge, Test ion charge. Discrete.
    ne : float
        m**-3, Electron density
    Te : float
        eV, Electron temperature
    logΛe : float
        Log(Λ) for ions colliding with electrons

    Outputs
    -------
    ts : float
       s, Slowing time of subthermal ions on eletrons
          Subthermal means that the ions are moving slower
          than electron thermal velocities.
    """
    def setup(self):
        # the constant is equal to
        # 4 π ε0² u² / (10^20 m⁻³ e⁴ 4 / (3 √π) (u / me) (me / (2 keV))^(3/2))
        # where
        # u is the atomic mass unit,
        # m is meters
        # e is the fundamental charge in C
        # me is the electron mass
        # eV is electron volts
        self.c = 0.19834312  # seconds
        # note that for n in m**-3 and Te in eV,
        # this constant c is 6.28e14 s as in Medley, 2004
        self.add_input("ne", units="n20", desc="Electron density")
        self.add_input("Te", units="keV", desc="Electron temperature")
        self.add_input("At", units="u", desc="Test particle mass")
        self.add_input("logΛe", desc="Collision log of test ion on e⁻")
        self.add_discrete_input("Zt", val=1, desc="Test particle charge")
        self.add_output("ts", units='s', desc="Slowing time of ions on e⁻")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        ne = inputs["ne"]
        Te = inputs["Te"]
        At = inputs["At"]
        Zt = discrete_inputs["Zt"]
        logLe = inputs["logΛe"]
        ts = self.c * At * Te**(3 / 2) / (ne * Zt**2 * logLe)
        outputs["ts"] = ts

    def setup_partials(self):
        self.declare_partials('ts', ['ne', 'Te', 'At', 'logΛe'])

    def compute_partials(self, inputs, J, discrete_inputs):
        ne = inputs["ne"]
        Te = inputs["Te"]
        At = inputs["At"]
        Zt = discrete_inputs["Zt"]
        logLe = inputs["logΛe"]
        J["ts", "ne"] = -self.c * At * Te**(3 / 2) / (ne**2 * Zt**2 * logLe)
        J["ts",
          "Te"] = (3 / 2) * self.c * At * Te**(1 / 2) / (ne * Zt**2 * logLe)
        J["ts", "At"] = self.c * Te**(3 / 2) / (ne * Zt**2 * logLe)
        J["ts", "logΛe"] = -self.c * At * Te**(3 / 2) / (ne * Zt**2 * logLe**2)


class AverageEnergyWhileSlowing(om.ExplicitComponent):
    r"""Average energy while slowing down

    1/τth \int_0^τth W(t) dt

    Inputs
    ------
    W/Wc : float
        Ratio of initial energy to critical energy
    Wc : float
        keV, Critical energy

    Outputs
    -------
    W-bar : float
        keV, Average energy while slowing
    """
    def setup(self):
        self.add_input(
            "W/Wc", desc="Ratio of initial energy to critical slowing energy")
        self.add_input("Wc", units='keV', desc="Critical slowing energy")
        self.add_output("Wbar",
                        units='keV',
                        desc="Average energy while slowing")

    def compute(self, inputs, outputs):
        wrat = inputs["W/Wc"]
        wc = inputs["Wc"]
        term1 = wc / (6 * np.log(1 + wrat**(3 / 2)))
        term2 = -4 * 3**(1 / 2) * pi
        arg = 1 / (1 + wrat**(3 / 2))
        term3 = 9 * (1 + wrat**(3 / 2))**(2 / 3) * hyp2f1(
            -2 / 3, -2 / 3, 1 / 3, arg)
        wbar = term1 * (term2 + term3)
        outputs["Wbar"] = wbar

    def setup_partials(self):
        self.declare_partials("Wbar", ["W/Wc", "Wc"])

    def compute_partials(self, inputs, J):
        wrat = inputs["W/Wc"]
        term1 = 1 / (6 * np.log(1 + wrat**(3 / 2)))
        term2 = -4 * 3**(1 / 2) * pi
        arg = 1 / (1 + wrat**(3 / 2))
        term3 = 9 * (1 + wrat**(3 / 2))**(2 / 3) * hyp2f1(
            -2 / 3, -2 / 3, 1 / 3, arg)
        dwbar_dwc = term1 * (term2 + term3)
        J["Wbar", "Wc"] = dwbar_dwc

        denom = 4 * (1 + wrat**(3 / 2)) * np.log(1 + wrat**(3 / 2))**2
        term1 = wrat**(1 / 2)
        # term2 = -term2
        # arg = 1 / (1 + wrat**(3/2))
        # term3 = -term3
        term4 = -6 * wrat * np.log(1 + wrat**(3 / 2))
        numer = -term1 * (term2 + term3 + term4)
        J["Wbar", "W/Wc"] = inputs["Wc"] * numer / denom


class CriticalSlowingEnergy(om.ExplicitComponent):
    r"""Critical energy for fast particles slowing down

    where energy is transferred equally to ions and electrons

    Inputs
    ------
    At : float
        u, test particle mass
    ni : array
        m**-3, ion densities
    Ai : array
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
        keV, Critical energy
    """
    def setup(self):
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


class CriticalSlowingEnergyRatio(om.ExplicitComponent):
    r"""Ratio of test particle energy to critical energy

    Inputs
    ------
    W : float
        keV, Test particle energy
    W_crit : float
        keV, Critical slowing energy

    Outputs
    -------
    W/Wc : float
        Ratio of the two
    """
    def setup(self):
        self.add_input("W", units="keV", desc="Test particle energy")
        self.add_input("W_crit", units="keV", desc="Critical slowing energy")
        self.add_output("W/Wc", desc="Ratio of energies")

    def compute(self, inputs, outputs):
        outputs["W/Wc"] = inputs["W"] / inputs["W_crit"]

    def setup_partials(self):
        self.declare_partials("W/Wc", ["W", "W_crit"])

    def compute_partials(self, inputs, J):
        J["W/Wc", "W"] = 1 / inputs["W_crit"]
        J["W/Wc", "W_crit"] = -inputs["W"] / inputs["W_crit"]**2


class FastParticleSlowing(om.Group):
    r"""
    Inputs
    ------
    S : float
        1/s, Fast particle source rate
    At : float
        u, Test particle mass
    Zt : int
        Fundamental charge, Test particle charge
    Wt : float
        keV, Test particle initial kinetic energy

    ne : float
        m**-3, electron density
    Te : float
        keV, electron temperature
    logΛe: float
        Coulomb logarithm

    ni : Array
        m**-3, ion densities
    Ai : Array
        u, ion masses
    Zi : Array
        Fundamental charges, ion charges

    Outputs
    -------
    τth : float
        s, Fast particle thermalization time
    f_i : float
        Fraction of energy which heats ions
    f_e : float
        Fraction of energy which heats electrons
    Wbar : float
        keV, Average energy while thermalizing from Wt
    Wfast : float
        MJ, Fast particle energy in plasma
    """
    def setup(self):
        self.add_subsystem(
            "Wcrit",
            CriticalSlowingEnergy(),
            promotes_inputs=["At", "ne", "Te", "ni", "Ai", "Zi"])
        self.add_subsystem("WcRat",
                           CriticalSlowingEnergyRatio(),
                           promotes_inputs=[("W", "Wt")])
        self.add_subsystem("slowingt",
                           SlowingTimeOnElectrons(),
                           promotes_inputs=["logΛe", "ne", "Te", "At", "Zt"])
        self.add_subsystem("thermalization",
                           SlowingThermalizationTime(),
                           promotes_outputs=["τth"])
        self.add_subsystem("heating",
                           FastParticleHeatingFractions(),
                           promotes_outputs=["*"])
        self.add_subsystem("averagew",
                           AverageEnergyWhileSlowing(),
                           promotes_outputs=["*"])
        self.add_subsystem("Wfast", om.ExecComp("Wfast = (Wbar) * tauth * S / mega",
            Wfast={"units":"MJ"},
            Wbar={"units":"J"},
            tauth={"units": "s"},
            mega={"value": mega},
            S={"units": "1/s"}), promotes_inputs=["Wbar", "tauth", "S"],
            promotes_outputs=["Wfast"])

        self.connect("Wcrit.W_crit", ["WcRat.W_crit", "averagew.Wc"])
        self.connect("WcRat.W/Wc",
                     ["thermalization.W/Wc", "heating.W/Wc", "averagew.W/Wc"])
        self.connect("slowingt.ts", ["thermalization.ts"])


if __name__ == "__main__":
    from openmdao.utils.assert_utils import assert_check_partials
    from scipy.constants import m_p
    prob = om.Problem()

    prob.model.add_subsystem('ivc',
                             om.IndepVarComp('ni', val=np.ones(3),
                                             units='n20'),
                             promotes_outputs=["*"])
    prob.model.add_subsystem('fps',
                             FastParticleSlowing(),
                             promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)

    prob.set_val("S", 6.24e20, units='1/s')
    prob.set_val("At", 2 * m_p, units='kg')
    prob.set_val("Zt", 1)
    prob.set_val("Wt", 500, units='keV')
    prob.set_val("ne", 1.06e20, units='m**-3')
    prob.set_val("Te", 9.2, units='keV')
    prob.set_val("logΛe", 17.37)
    prob.set_val("ni", np.array([0.424e20, 0.424e20, 0.0353e20]), units='m**-3')
    prob.set_val("Ai", [2, 3, 12], units='u')
    prob.set_val("Zi", [1, 1, 6])

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)

    prob.run_driver()
    all_outputs = prob.model.list_outputs(values=True, print_arrays=True)
