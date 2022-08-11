import faroes.units  # noqa: F401
from faroes.plasmaformulary import CoulombLogarithmElectrons
from faroes.configurator import UserConfigurator

import openmdao.api as om
from openmdao.utils.units import unit_conversion

from scipy.constants import pi, electron_mass
from scipy.special import hyp2f1
import numpy as np


class SlowingThermalizationTime(om.ExplicitComponent):
    r"""Time to slow down to zero velocity

    .. math::

       \tau_\mathrm{th} = \frac{t_s}{3}
           \log\left(1 +
           \left(\frac{W}{W_\mathrm{crit}}\right)^{3/2}\right)

    Here, :math:`t_s` is the ion-electron slowing down time given
    by Spitzer :footcite:p:`spitzer_physics_2006`.


    Inputs
    ------
    W/Wc : float
        Initial beam energy to critical energy
    ts : float
        s, Ion-electron velocity slowing down time

    Outputs
    -------
    τth : float
        s, Thermalization time
        (time to slow to zero average velocity, in this model)

    References
    ----------
    :footcite:t:`stix_heating_1972`
    """
    def setup(self):
        self.add_input("W/Wc", desc="Initial beam energy / critical energy")
        self.add_input("ts",
                       units="s",
                       desc="Velocity slowing time of ions on e⁻")
        self.add_output("τth",
                        lower=0,
                        units="s",
                        desc="Fast particle thermalization time")

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

    This is the exact solution to an integral

    .. math::

       G &= \frac{W_c}{W} \int_0^{W/W_c} \frac{dy}{1 + y^{3/2}}

         &= \, _2F_1\left(\frac{2}{3},1;\frac{5}{3};-(W/W_c)^{3/2}\right)

    where :math:`W_c` is the critical slowing energy. The fraction of the
    energy which goes to ions :math:`f_i = G` and the fraction to electrons
    is :math:`f_e = 1 - G`.

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

    Notes
    -----

    In Mathematica:

    ``G = wc/w Integrate[1/(1 + y^(3/2)), {y, 0, w/wc}]``

    References
    ----------
    Equation (17) of :footcite:t:`stix_heating_1972`.
    """
    def setup(self):
        self.add_input("W/Wc", desc="Initial beam energy / critical energy")
        self.add_output("f_i",
                        val=0.5,
                        desc="Fraction of fast particle energy to ions")
        self.add_output("f_e",
                        val=0.5,
                        desc="Fraction of fast particle energy to electrons")

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

    The velocity slowing time is defined such that

    .. math::

       du/dt = - u / \tau_s.

    Where :math:`u` is the average ion speed.  The energy slowing time is

    .. math::

       dW/dt = -W / \tau_{s,w}.

    Since :math:`(d/dt) (1/2)m u^2 = m u du/dt` it is easy to show that
    :math:`\tau_{s} = 2 \tau_{s,w}`.

    .. math::

       \tau_s = c \frac{A_b (T_e / \mathrm{keV})^{3/2}}
           {Z^2 (n_e / (10^{20} \mathrm{m}^{-3}) \log{\Lambda_e}}

    Inputs
    ------
    At : float
        u, Test particle mass
    Zt : int
        e, Test ion charge
    ne : float
        n20, Electron density
    Te : float
        eV, Electron temperature
    logΛe : float
        Log(Λ) for ions colliding with electrons

    Outputs
    -------
    ts : float
       s, Velocity slowing time of subthermal ions on electrons.
       Subthermal means that the ions are moving slower
       than electron thermal velocities.
    tsw : float
       s, Energy slowing time of subthermal ions on electrons.

    References
    ----------
    Menard uses Equation (2) of Medley :footcite:t:`medley_mhd-induced_2004`.
    That version includes a numerical constant, 6.27e14.
    Medley cites Spitzer :footcite:p:`spitzer_physics_2006`.

    See Equation (13.72) of Bellan :footcite:p:`bellan_fundamentals_2006`.
    for comparison. The Bellan equation has a term :math:`(1 + m_T / m_e)`
    but here the test particles are ions, so their masses are much larger
    than the electron masses; that term is not present in Medley.
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
        # this constant c is 6.28e14 s as in Medley, 2004
        # where it equals
        # 4 π ε0² u² / (m⁻³ e⁴ 4 / (3 √π) (u / me) (me / (2 eV))^(3/2)).
        # to highlight the differences: 10^20 m⁻³ → m⁻³ and keV → eV.
        self.add_input("ne", units="n20", desc="Electron density")
        self.add_input("Te", units="keV", desc="Electron temperature")
        self.add_input("At", units="u", desc="Test particle mass")
        self.add_input("logΛe", desc="Collision log of test ion on e⁻")
        self.add_input("Zt", val=1, desc="Test particle charge")
        self.add_output("ts",
                        units="s",
                        desc="Velocity slowing time of ions on e⁻")
        self.add_output("tsw",
                        units="s",
                        desc="Energy slowing time of ions on e⁻")

    def compute(self, inputs, outputs):
        ne = inputs["ne"]
        Te = inputs["Te"]
        At = inputs["At"]
        Zt = inputs["Zt"]
        logLe = inputs["logΛe"]
        ts = self.c * At * Te**(3 / 2) / (ne * Zt**2 * logLe)
        outputs["ts"] = ts
        outputs["tsw"] = ts / 2

    def setup_partials(self):
        self.declare_partials('ts', ['ne', 'Te', 'At', 'logΛe', 'Zt'])
        self.declare_partials('tsw', ['ne', 'Te', 'At', 'logΛe', 'Zt'])

    def compute_partials(self, inputs, J):
        ne = inputs["ne"]
        Te = inputs["Te"]
        At = inputs["At"]
        Zt = inputs["Zt"]
        logLe = inputs["logΛe"]
        J["ts", "ne"] = -self.c * At * Te**(3 / 2) / (ne**2 * Zt**2 * logLe)
        J["ts",
          "Te"] = (3 / 2) * self.c * At * Te**(1 / 2) / (ne * Zt**2 * logLe)
        J["ts", "At"] = self.c * Te**(3 / 2) / (ne * Zt**2 * logLe)
        J["ts", "logΛe"] = -self.c * At * Te**(3 / 2) / (ne * Zt**2 * logLe**2)
        J["ts", "Zt"] = -2 * self.c * At * Te**(3 / 2) / (ne * Zt**3 * logLe)
        J["tsw", "ne"] = J["ts", "ne"] / 2
        J["tsw", "Te"] = J["ts", "Te"] / 2
        J["tsw", "At"] = J["ts", "At"] / 2
        J["tsw", "logΛe"] = J["ts", "logΛe"] / 2
        J["tsw", "Zt"] = J["ts", "Zt"] / 2


class AverageEnergyWhileSlowing(om.ExplicitComponent):
    r"""Average energy while slowing down

    .. math::

        \bar{W} = \frac{1}{\tau_\mathrm{th}} \int_0^{\tau_{th}} W(t) \; dt

    With :math:`W_r \equiv W/W_c`,

    .. math::

        \bar{W} = \frac{W_c}{6 \log(1 + W_r)} \left(-4\cdot\, 3^{1/2} \pi +
          9(1 + W_r^{3/2})^{2/3}
             \, _2F_1\left(-\frac{2}{3}, -\frac{2}{3}; \frac{1}{3};
             \frac{1}{1 + W_r^{3/2}} \right)\right)

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
    SLOW_RATIO = 1e-1
    SLOW_ERR = """Fast particles are subthermal. Probably something has gone
    wrong upstream, like the Te being extraordinarily high."""

    def setup(self):
        self.add_input(
            "W/Wc", desc="Ratio of initial energy to critical slowing energy")
        self.add_input("Wc", units='keV', desc="Critical slowing energy")
        Wbar_ref = 100
        self.add_output("Wbar",
                        units='keV',
                        ref=Wbar_ref,
                        desc="Average energy while slowing")

    def compute(self, inputs, outputs):
        wrat = inputs["W/Wc"]
        if wrat < self.SLOW_RATIO:
            raise om.AnalysisError(self.SLOW_ERR)

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
        if wrat < self.SLOW_RATIO:
            raise om.AnalysisError(self.SLOW_ERR)

        term1 = 1 / (6 * np.log(1 + wrat**(3 / 2)))
        term2 = -4 * 3**(1 / 2) * pi
        arg = 1 / (1 + wrat**(3 / 2))
        term3 = 9 * (1 + wrat**(3 / 2))**(2 / 3) * hyp2f1(
            -2 / 3, -2 / 3, 1 / 3, arg)
        dwbar_dwc = term1 * (term2 + term3)
        J["Wbar", "Wc"] = dwbar_dwc

        denom = 4 * (1 + wrat**(3 / 2)) * np.log(1 + wrat**(3 / 2))**2
        term1 = wrat**(1 / 2)
        term4 = -6 * wrat * np.log(1 + wrat**(3 / 2))
        numer = -term1 * (term2 + term3 + term4)
        J["Wbar", "W/Wc"] = inputs["Wc"] * numer / denom


class StixCriticalSlowingEnergy(om.ExplicitComponent):
    r"""Critical energy for fast particles slowing down

    This is the energy at which the energy lost is transferred equally to ions
    and electrons, as computed by :footcite:t:`stix_heating_1972`.

    .. math::

        \alpha' &= A_t \sum_i n_i Z_i^2 / A_i

        \beta' &= \frac{4}{3 \pi^{1/2}} n_e

        W_\mathrm{crit} &= T_e \left(\frac{m_T}{m_e}\right)^{1/3}
            \left(\frac{\alpha'}{\beta'}\right)^{2/3}

    Inputs
    ------
    At : float
        u, test particle mass
    ni : array
        n20, ion densities
    Ai : array
        u, ion masses
    Zi : array
        e, ion charges
    ne : float
        n20, electron density
    Te : float
        eV, electron temperature

    Outputs
    -------
    W_crit : float
        keV, Critical energy for slowing ions
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

        W_crit_ref = 100
        self.add_output('W_crit',
                        units='keV',
                        lower=0,
                        ref=W_crit_ref,
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
        α = np.sum(ni * zi**2 * (At / Ai))
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
        α = np.sum(ni * zi**2 * (At / Ai))
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

        dαdni = (At / Ai) * zi**2
        numer = (2 / 3) * mass_scale * Te * dαdni
        denom = (β**(2 / 3) * α**(1 / 3))
        J["W_crit", "ni"] = numer / denom

        dαdzi = 2 * (At / Ai) * ni * zi
        numer = (2 / 3) * mass_scale * Te * dαdzi
        J["W_crit", "Zi"] = numer / denom

        dαdAi = -(At / Ai**2) * ni * zi**2
        numer = (2 / 3) * mass_scale * Te * dαdAi
        J["W_crit", "Ai"] = numer / denom


class BellanCriticalSlowingEnergy(om.ExplicitComponent):
    r"""Critical energy for fast particles slowing down

    where energy is transferred equally to ions and electrons

    .. math::

        \alpha' &= \sum_i n_i Z_i^2 (1 + A_t / A_i)

        \beta' &= \frac{4}{3 \pi^{1/2}} n_e

        W_\mathrm{crit} &= T_e \left(\frac{m_T}{m_e}\right)^{1/3}
            \left(\frac{\alpha'}{\beta'}\right)^{2/3}


    Inputs
    ------
    At : float
        u, test particle mass
    ni : array
        n20, ion densities
    Ai : array
        u, ion masses
    Zi : array
        e, ion charges
    ne : float
        n20, electron density
    Te : float
        eV, electron temperature

    Outputs
    -------
    W_crit : float
        keV, Critical energy

    Notes
    -----

    Bellan :footcite:p:`bellan_fundamentals_2006` does not explicitly
    provide a formula for the critical slowing energy, but it can be
    derived from Equation 13.72. The major difference from the treatment
    of :footcite:t:`stix_heating_1972` is that the summation here called
    :math:`\alpha'` is over :math:`n_i Z_i^2 (1 + A_t/A_i)`,
    whereas Stix sums :math:`n_i Z_i^2 (A_t/A_i)`. This term is a reduced mass
    :math:`\mu` and can be traced back to the collision operator. I'm not sure
    why it's not present in Stix's treatment.
    Stix cites Sivukhin, Equation 8.1.

    References
    ----------

       Sivukhin, D. V. (1966).
       Reviews of Plasma Physics (M. A. Leontovich, Ed.)
       Consultants Bureau, New York, Vol. 4, p.93, Equation 8.1

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

        W_crit_ref = 100
        self.add_output('W_crit',
                        units='keV',
                        lower=0,
                        ref=W_crit_ref,
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

    This module simply combines the two inputs into a single
    quantity representing their ratio, which is useful to simplify various
    derivative calculations.

    .. math:: \left[\frac{W}{W_c}\right] = W / W_c

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
        self.add_output(
            "W/Wc",
            lower=0,
            desc="Ratio of initial energy to critical slowing energy")

    def compute(self, inputs, outputs):
        outputs["W/Wc"] = inputs["W"] / inputs["W_crit"]

    def setup_partials(self):
        self.declare_partials("W/Wc", ["W", "W_crit"])

    def compute_partials(self, inputs, J):
        J["W/Wc", "W"] = 1 / inputs["W_crit"]
        J["W/Wc", "W_crit"] = -inputs["W"] / inputs["W_crit"]**2


class FastParticleSlowing(om.Group):
    r"""Top-level group for fast particles

    After calculating the critical fast particle energy it can be scaled by a
    multiplier before proceeding with further calculations.

    .. math::

        W_\mathrm{crit} = \mathrm{scale} \; W_{\mathrm{crit},0}

    Given the source rate :math:`S` for a given group of fast particles
    the group also calculates the total fast-particle stored energy
    from that group.

    .. math::

        W_\mathrm{fast} = \bar{W}\, \tau_\mathrm{th}\, S

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Inputs
    ------
    S : float
        1/s, Fast particle source rate
    At : float
        u, Test particle mass
    Zt : int
        e, Test particle charge
    Wt : float
        keV, Test particle initial kinetic energy

    ne : float
        n20, electron density
    Te : float
        keV, electron temperature

    ni : Array
        n20, ion densities
    Ai : Array
        u, ion masses
    Zi : Array
        e, ion charges

    Outputs
    -------
    τs : float
        s, Slowing time on electrons
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
    logΛe: float
        Coulomb logarithm for electrons

    Notes
    -----
    Inputs from the configuration file:

    method:
       "Stix" or "Menard" or "Bellan"
       This controls the calculation method for the critical slowing energy.
       "Menard" is an alias for "Stix" here.

    Te profile peaking factor for W_crit:
       Acts as a scaling factor for W_crit
    """
    BAD_METHOD_STR = """
    Unknown method '%s' specified for fast particle slowing.
    Valid methods are %s """
    SUPPORTED_METHODS = ["Bellan", "Stix", "Menard"]

    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

    def setup(self):
        config = self.options["config"]
        acc = config.accessor(["h_cd", "NBI", "fast-ion slowing"])
        method = acc(["method"])

        if method is None or method == "default":
            method = acc(["default"])

        if method == "Bellan":
            cse = BellanCriticalSlowingEnergy()
        elif method in ["Menard", "Stix"]:
            cse = StixCriticalSlowingEnergy()
        else:
            raise ValueError(self.BAD_METHOD_STR %
                             (method, self.SUPPORTED_METHODS))

        acc = self.options["config"].accessor(["plasma"])
        scale = acc(["Te profile peaking factor for W_crit"])

        self.add_subsystem(
            "Wcrit0",
            cse,
            promotes_inputs=["At", "ne", "Te", "ni", "Ai", "Zi"])
        self.add_subsystem(
            "Wcrit",
            om.ExecComp("W_crit = scale * Wcrit0",
                        W_crit={
                            "units": "keV",
                            'desc': "Adjusted critical slowing energy"
                        },
                        Wcrit0={
                            "units": "keV",
                            'desc': "Critical slowing energy"
                        },
                        scale={
                            "val": scale,
                            'desc': "Critical slowing energy adj. factor"
                        }))
        self.connect("Wcrit0.W_crit", "Wcrit.Wcrit0")

        self.add_subsystem("logCoulombEl",
                           CoulombLogarithmElectrons(),
                           promotes_inputs=["ne", "Te"],
                           promotes_outputs=["logΛe"])
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
        self.add_subsystem(
            "Wfast",
            om.ExecComp("Wfast = (Wbar) * tauth * S / 10**6",
                        Wfast={
                            "units": "MJ",
                            'desc': "Fast particle thermal energy"
                        },
                        Wbar={
                            "units": "J",
                            'desc':
                            "Average energy of fast particle while slowing"
                        },
                        tauth={
                            "units": "s",
                            'desc': "Thermalization time"
                        },
                        S={
                            "units": "1/s",
                            'desc': "Fast particle source rate"
                        }),
            promotes_inputs=["Wbar", ("tauth", "τth"), "S"],
            promotes_outputs=["Wfast"])

        self.connect("Wcrit.W_crit", ["WcRat.W_crit", "averagew.Wc"])
        self.connect("WcRat.W/Wc",
                     ["thermalization.W/Wc", "heating.W/Wc", "averagew.W/Wc"])
        self.connect("slowingt.ts", ["thermalization.ts"])
        self.set_input_defaults("Te", units="keV", val=10)


if __name__ == "__main__":
    from openmdao.utils.assert_utils import assert_check_partials
    from scipy.constants import m_p
    prob = om.Problem()
    uc = UserConfigurator()

    prob.model.add_subsystem('ivc',
                             om.IndepVarComp('ni', val=np.ones(3),
                                             units='n20'),
                             promotes_outputs=["*"])
    prob.model.add_subsystem('fps',
                             FastParticleSlowing(config=uc),
                             promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)

    prob.set_val("S", 6.24e20, units='1/s')
    prob.set_val("At", 2 * m_p, units='kg')
    prob.set_val("Zt", 1)
    prob.set_val("Wt", 500, units='keV')
    prob.set_val("ne", 1.06, units="n20")
    prob.set_val("Te", 9.2, units='keV')
    prob.set_val("ni",
                 np.array([0.424e20, 0.424e20, 0.0353e20]),
                 units='m**-3')
    prob.set_val("Ai", [2, 3, 12], units='u')
    prob.set_val("Zi", [1, 1, 6])

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)

    prob.run_driver()
    all_inputs = prob.model.list_inputs(val=True,
                                        print_arrays=True,
                                        units=True,
                                        desc=True)
    all_outputs = prob.model.list_outputs(val=True,
                                          print_arrays=True,
                                          units=True,
                                          desc=True)
