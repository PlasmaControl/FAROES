import openmdao.api as om
from faroes.configurator import UserConfigurator
from plasmapy.particles import deuteron, triton
from plasmapy.particles import Particle
from scipy.constants import mega, kilo, atm, eV, electron_mass, pi

from faroes.fusionreaction import SimpleRateCoeff, VolumetricThermalFusionRate


class MainIonMix(om.ExplicitComponent):
    r"""Main ion species mix

    Represents the mix of main ion species: D and T.
    This does not include any He ash or impurities in the plasma.

    Inputs
    ------
    f_D: float
        Fraction of deuterium in main ions. Defaults to 0.5.

    Outputs
    -------
    f_T: float
        Fraction of tritium in main ions. Defaults to 0.5.
    A : float
        Averaged ion mass number
    m : float
        kg, Averaged main ion mass
    """
    def setup(self):
        self.add_input("f_D", val=0.5, desc="Fraction of D in main ions")
        self.add_output("f_T", val=0.5, desc="Fraction of T in main ions")
        self.add_output("A",
                        val=2.5,
                        lower=1,
                        upper=3,
                        desc="Averaged main ion mass number")
        self.add_output("m", units='kg', desc="Averaged main ion mass")

    def mass_kg(self, p):
        """get mass in kg of Particle p
        """
        return p.mass.value

    def compute(self, inputs, outputs):
        f_D = inputs["f_D"]
        f_T = 1 - f_D
        outputs["f_T"] = f_T
        a_D = deuteron.mass_number
        a_T = triton.mass_number
        A = f_D * a_D + f_T * a_T
        outputs["A"] = A

        m_D = self.mass_kg(deuteron)
        m_T = self.mass_kg(triton)
        m = f_D * m_D + f_T * m_T
        outputs["m"] = m

    def setup_partials(self):
        self.declare_partials("f_T", "f_D", val=-1)
        a_D = deuteron.mass_number
        a_T = triton.mass_number
        self.declare_partials("A", "f_D", val=(a_D - a_T))
        m_D = self.mass_kg(deuteron)
        m_T = self.mass_kg(triton)
        self.declare_partials("m", "f_D", val=(m_D - m_T))


class ZeroDPlasmaProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("ZeroDPlasmaProperties requries a config file")
        config = self.options['config'].accessor(['plasma'])
        t_rat = config(['Ti/Te'])
        self.add_output('Ti/Te',
                        val=t_rat,
                        desc="Ion-electron temperature ratio")

        fus_enhancement = config(['Fusion enhancement from p-peaking'])
        self.add_output('P_fus enhancement from p-peaking',
                        val=fus_enhancement,
                        desc="Fusion enhancement from p-peaking")

        imp_model = config(['impurities', 'model'])
        if imp_model != "oneimpurity":
            raise ValueError("Only the One-Impurity model is implemented")

        imp = config(['impurities', imp_model])
        z_eff = imp["Z_eff"]
        self.add_output('Z_eff', val=z_eff, desc="Effective ion charge")
        impurity = Particle(imp["species"])
        self.add_output('Z_imp',
                        val=impurity.integer_charge,
                        desc='Impurity charge')
        self.add_output('A_imp',
                        val=impurity.mass_number,
                        desc="Impurity mass number")
        self.add_output('m_imp',
                        val=impurity.mass.value,
                        units='kg',
                        desc="Impurity mass")

        f_GW = config(['Greenwald fraction'])
        self.add_output('f_GW', val=f_GW, desc="Greenwald fraction")


class ZeroDPlasmaDensities(om.ExplicitComponent):
    r"""
    Inputs
    ------
    n_e : float
        m**-3, Electron density
    Z_eff : float
        Effective ion charge.
        Default: 2.0
    Z_imp : float
        Main impurity charge
    m_imp : float
        kg, Main impurity mass
    m_main : float
        kg, Main ion mass
    f_D : float
        Fraction of main plasma ions which are deuterium.
        Defaults to 0.5.

    Outputs
    -------
    n_main_i/ne : float
        Ratio of main ion density to electron density
    n_imp/ne : float
        Ratio of impurity density to electron density
    ni/ne : float
        Ratio of ion density to electron density
    n_main_i : float
        m**-3, Main ion density
    n_D : float
        m**-3, deuterium density
    n_T : float
        m**-3, deuterium density
    Z_ave : float
        Average ion charge
    """
    def setup(self):
        self.add_input("f_D", val=0.5)
        self.add_input("m_main_i", units="kg", desc="Main ion mass")
        self.add_input("Z_eff", val=2.0)
        self.add_input("Z_imp", desc="Impurity charge state")
        self.add_input("m_imp", units="kg", desc="Impurity mass")
        self.add_input("n_e", units="m**-3", desc="Electron density")

        self.add_output("n_main_i/ne",
                        lower=0,
                        desc="Ratio of main ion density to ne")
        self.add_output("n_imp/ne",
                        lower=0,
                        desc="Ratio of impurities to electrons")
        self.add_output("n_imp",
                        units="m**-3",
                        lower=0,
                        desc="Impurity density")
        self.add_output("ni/ne",
                        lower=0,
                        desc="Ratio of impurities to electrons")
        self.add_output("n_main_i",
                        units="m**-3",
                        ref=1e20,
                        lower=0,
                        desc="Main ion density")
        self.add_output("Z_ave", desc="Average ion charge")
        self.add_output("n_D", units="m**-3", desc="Deuterium ion density")
        self.add_output("n_T", units="m**-3", desc="Tritium ion density")
        self.add_output("ρ", units="kg/m**3", desc="Plasma mass density")

    def compute(self, inputs, outputs):
        f_D = inputs["f_D"]
        n_e = inputs["n_e"]
        z_eff = inputs["Z_eff"]
        z_imp = inputs["Z_imp"]
        m_imp = inputs["m_imp"]
        n_main_i_frac = (z_imp - z_eff) / (z_imp - 1)
        n_imp_frac = (1 - n_main_i_frac) / z_imp
        n_ion_frac = n_main_i_frac + n_imp_frac
        n_main_i = n_main_i_frac * n_e
        m_main_i = inputs["m_main_i"]

        outputs["n_main_i/ne"] = n_main_i_frac
        outputs["n_imp/ne"] = n_imp_frac
        n_imp = n_imp_frac * n_e
        outputs["n_imp"] = n_imp
        outputs["ni/ne"] = n_ion_frac
        outputs["n_main_i"] = n_main_i
        outputs["Z_ave"] = 1 / n_ion_frac
        outputs["n_D"] = n_main_i * f_D
        outputs["n_T"] = n_main_i * (1 - f_D)

        ρ = electron_mass * n_e + m_main_i * n_main_i + m_imp * n_imp
        outputs["ρ"] = ρ

    def setup_partials(self):
        self.declare_partials("n_main_i/ne", ["Z_imp", "Z_eff"])
        self.declare_partials("n_imp/ne", ["Z_imp", "Z_eff"])
        self.declare_partials("n_imp", ["Z_imp", "Z_eff", 'n_e'])
        self.declare_partials("ni/ne", ["Z_imp", "Z_eff"])
        self.declare_partials("n_main_i", ["Z_imp", "Z_eff", "n_e"])
        self.declare_partials("Z_ave", ["Z_imp", "Z_eff"])
        self.declare_partials("n_D", ["Z_imp", "Z_eff", "n_e", "f_D"])
        self.declare_partials("n_T", ["Z_imp", "Z_eff", "n_e", "f_D"])
        self.declare_partials("ρ",
                              ["m_main_i", "Z_imp", "Z_eff", "m_imp", "n_e"])

    def compute_partials(self, inputs, J):
        f_D = inputs["f_D"]
        n_e = inputs["n_e"]
        z_eff = inputs["Z_eff"]
        z_imp = inputs["Z_imp"]
        m_imp = inputs["m_imp"]
        m_main_i = inputs["m_main_i"]
        n_main_i_frac = (z_imp - z_eff) / (z_imp - 1)
        n_main_i = n_main_i_frac * n_e
        n_imp_frac = (1 - n_main_i_frac) / z_imp
        n_imp = n_imp_frac * n_e

        J["n_main_i/ne", "Z_imp"] = (z_eff - 1) / (z_imp - 1)**2
        J["n_main_i/ne", "Z_eff"] = 1 / (1 - z_imp)
        denom = (z_imp * (z_imp - 1))
        J["n_imp/ne", "Z_imp"] = -(z_eff - 1) * (2 * z_imp - 1) / denom**2
        J["n_imp/ne", "Z_eff"] = 1 / denom
        J["n_imp", "Z_imp"] = -n_e * (z_eff - 1) * (2 * z_imp - 1) / denom**2
        J["n_imp", "Z_eff"] = n_e / denom
        J["n_imp", "n_e"] = n_imp_frac

        J["ni/ne", "Z_imp"] = (z_eff - 1) / z_imp**2
        J["ni/ne", "Z_eff"] = -1 / z_imp
        J["n_main_i", "n_e"] = (z_imp - z_eff) / (z_imp - 1)
        J["n_main_i", "Z_imp"] = n_e * (z_eff - 1) / (z_imp - 1)**2
        J["n_main_i", "Z_eff"] = -n_e / (z_imp - 1)
        J["Z_ave", "Z_imp"] = (1 - z_eff) / (1 + z_imp - z_eff)**2
        J["Z_ave", "Z_eff"] = z_imp / (1 + z_imp - z_eff)**2

        J["n_D", "n_e"] = f_D * J["n_main_i", "n_e"]
        J["n_D", "Z_imp"] = f_D * J["n_main_i", "Z_imp"]
        J["n_D", "Z_eff"] = f_D * J["n_main_i", "Z_eff"]
        J["n_D", "f_D"] = n_main_i
        J["n_T", "n_e"] = (1 - f_D) * J["n_main_i", "n_e"]
        J["n_T", "Z_imp"] = (1 - f_D) * J["n_main_i", "Z_imp"]
        J["n_T", "Z_eff"] = (1 - f_D) * J["n_main_i", "Z_eff"]
        J["n_T", "f_D"] = -n_main_i

        J["ρ", "n_e"] = (electron_mass + m_main_i * J["n_main_i", "n_e"] +
                         m_imp * J["n_imp", "n_e"])
        J["ρ", "Z_imp"] = (m_main_i * J["n_main_i", "Z_imp"] +
                           m_imp * J["n_imp", "Z_imp"])
        J["ρ", "Z_eff"] = (m_main_i * J["n_main_i", "Z_eff"] +
                           m_imp * J["n_imp", "Z_eff"])
        J["ρ", "m_main_i"] = n_main_i
        J["ρ", "m_imp"] = n_imp


class ZeroDPlasmaStoredEnergy(om.ExplicitComponent):
    r"""
    Inputs
    ------
    V : float
        m**3, plasma volume
    τ_th : float
        s, Thermal energy confinement time
    P_loss : float
        MW, Thermal gradient loss power
    W_fast_NBI : float
        MJ, Energy of NBI fast ions in plasma
    W_fast_α : float
        MJ, Energy of fast α particles in plasma

    Outputs
    -------
    W_th : float
        MJ, Thermal particle energy
    W_fast : float
        MJ, Total fast particle energy
    W_tot : float
        MJ, Total energy of plasma
    <p_th> : float
        kPa, Averaged thermal particle pressure
    <p_tot> : float
        kPa, Averaged total pressure, thermal + fast
    p τE : float
        atm*s, pressure times energy confinement time
    thermal pressure fraction : float
        Fraction of pressure from thermal particles.
    """
    def setup(self):
        self.add_input("V", units="m**3", desc="Plasma volume")
        self.add_input("τ_th",
                       units="s",
                       desc="Thermal energy confinement time")
        self.add_input("P_loss",
                       units="MW",
                       desc="Thermal gradient loss power")
        self.add_input("W_fast_NBI",
                       units="MJ",
                       desc="Energy of NBI fast ions")
        self.add_input("W_fast_α",
                       units="MJ",
                       desc="Energy of fast α particles")

        self.add_output("W_th", units="MJ", desc="Thermal particle energy")
        self.add_output("W_fast",
                        units="MJ",
                        desc="Total fast particle energy")
        self.add_output("W_tot", units="MJ", desc="Total particle energy")
        self.add_output("<p_th>",
                        lower=0,
                        units="kPa",
                        desc="Averaged thermal particle pressure")
        self.add_output("<p_tot>",
                        lower=0,
                        units="kPa",
                        desc="Averaged total pressure")
        self.add_output("thermal pressure fraction",
                        lower=0,
                        upper=1,
                        desc="Fraction of pressure from thermal particles")
        self.add_output("p τE",
                        lower=0,
                        units="atm*s",
                        desc="Pressure * energy confinement time")

    def compute(self, inputs, outputs):
        V = inputs["V"]
        W_fast_NBI = inputs["W_fast_NBI"]
        W_fast_α = inputs["W_fast_α"]
        τ_th = inputs["τ_th"]
        P_loss = inputs["P_loss"]
        W_th = P_loss * τ_th
        outputs["W_th"] = W_th
        W_fast = W_fast_α + W_fast_NBI
        outputs["W_fast"] = W_fast
        W_tot = W_fast + W_th
        outputs["W_tot"] = W_tot
        p_th = (W_th / V) * (2 / 3) * (mega / kilo)
        outputs["<p_th>"] = p_th
        p_tot = (W_tot / V) * (2 / 3) * (mega / kilo)
        outputs["<p_tot>"] = p_tot
        outputs["thermal pressure fraction"] = p_th / p_tot
        outputs["p τE"] = p_tot * τ_th * kilo / atm

    def setup_partials(self):
        self.declare_partials("W_th", ["τ_th", "P_loss"])
        self.declare_partials("W_fast", ["W_fast_NBI", "W_fast_α"], val=1)
        self.declare_partials("W_tot", ["W_fast_NBI", "W_fast_α"], val=1)
        self.declare_partials("W_tot", ["τ_th", "P_loss"], val=1)
        self.declare_partials("<p_th>", ["V", "P_loss", "τ_th"])
        self.declare_partials(
            "<p_tot>", ["V", "P_loss", "τ_th", "W_fast_α", "W_fast_NBI"])
        self.declare_partials("thermal pressure fraction", "*", method="cs")
        self.declare_partials("p τE", "*", method="cs")

    def compute_partials(self, inputs, J):
        V = inputs["V"]
        W_fast_NBI = inputs["W_fast_NBI"]
        W_fast_α = inputs["W_fast_α"]
        W_fast = W_fast_α + W_fast_NBI
        τ_th = inputs["τ_th"]
        P_loss = inputs["P_loss"]
        W_tot = W_fast + P_loss * τ_th
        J["W_th", "τ_th"] = P_loss
        J["W_th", "P_loss"] = τ_th
        J["W_tot", "τ_th"] = P_loss
        J["W_tot", "P_loss"] = τ_th
        J["<p_th>", "P_loss"] = kilo * (2 / 3) * τ_th / V
        J["<p_th>", "τ_th"] = kilo * (2 / 3) * P_loss / V
        J["<p_th>", "V"] = -(2 / 3) * kilo * P_loss * τ_th / V**2

        J["<p_tot>", "P_loss"] = kilo * (2 / 3) * τ_th / V
        J["<p_tot>", "τ_th"] = kilo * (2 / 3) * P_loss / V
        J["<p_tot>", "V"] = -(2 / 3) * kilo * (W_tot) / V**2
        J["<p_tot>", "W_fast_NBI"] = (2 / 3) * kilo / V
        J["<p_tot>", "W_fast_α"] = (2 / 3) * kilo / V


class ZeroDPlasmaPressures(om.ExplicitComponent):
    r"""Zero-D plasma pressures
    Inputs
    ------
    <p_th> : float
        kPa, Averaged thermal particle pressure
    Ti/Te : float
        ion-electron temperature ratio.
        Defaults: 1.0.
    Z_ave : float
        Average Z of plasma ions

    Outputs
    -------
    <p_e> : float
        kPa, Averaged electron pressure
    <p_ion> : float
        kPa, Averaged ion pressure
    """
    def setup(self):
        self.add_input("Ti/Te", val=1.0)
        self.add_input("Z_ave")
        self.add_input("<p_th>",
                       units="kPa",
                       desc="Averaged thermal particle pressure")

        self.add_output("<p_e>",
                        lower=0,
                        units="kPa",
                        desc="Averaged electron pressure")
        self.add_output("<p_i>",
                        lower=0,
                        units="kPa",
                        desc="Averaged ion pressure")

    def compute(self, inputs, outputs):
        p_th = inputs["<p_th>"]
        T_rat = inputs["Ti/Te"]
        z_ave = inputs["Z_ave"]
        p_e = p_th / (1 + T_rat / z_ave)
        outputs["<p_e>"] = p_e
        p_i = p_th - p_e
        outputs["<p_i>"] = p_i

    def setup_partials(self):
        self.declare_partials("<p_e>", ["<p_th>", "Ti/Te", "Z_ave"])
        self.declare_partials("<p_i>", ["<p_th>", "Ti/Te", "Z_ave"])

    def compute_partials(self, inputs, J):
        p_th = inputs["<p_th>"]
        T_rat = inputs["Ti/Te"]
        z_ave = inputs["Z_ave"]
        J["<p_e>", "<p_th>"] = 1 / (1 + T_rat / z_ave)
        J["<p_e>", "Ti/Te"] = -p_th / (z_ave * (1 + T_rat / z_ave)**2)
        J["<p_e>", "Z_ave"] = p_th * T_rat / (T_rat + z_ave)**2

        J["<p_i>", "Z_ave"] = -p_th * T_rat / (T_rat + z_ave)**2
        J["<p_i>", "<p_th>"] = T_rat / (T_rat + z_ave)
        J["<p_i>", "Ti/Te"] = p_th * z_ave / (T_rat + z_ave)**2


class ZeroDPlasmaTemperatures(om.ExplicitComponent):
    r"""Zero-D plasma pressures and temperature
    Inputs
    ------
    <n_e> : float
        m**-3, Average electron density
    ni/ne : float
        Ratio of ion density to electron density
    <p_e> : float
        kPa, Averaged electron pressure
    <p_ion> : float
        kPa, Averaged ion pressure

    Outputs
    -------
    <T_e> : float
        keV, Averaged electron temperature
    <T_i> : float
        keV, Averaged ion temperature
    """
    def setup(self):
        self.add_input("<n_e>", units="m**-3", desc="Average electron density")
        self.add_input("ni/ne", desc="Ratio of ions to electrons")
        self.add_input("<p_e>", units="kPa", desc="Averaged electron pressure")
        self.add_input("<p_i>", units="kPa", desc="Averaged ion pressure")

        self.add_output("<T_e>",
                        lower=0,
                        units="keV",
                        desc="Averaged electron temperature")
        self.add_output("<T_i>",
                        lower=0,
                        units="keV",
                        desc="Averaged ion temperature")

    def compute(self, inputs, outputs):
        p_e = inputs["<p_e>"]
        p_i = inputs["<p_i>"]
        n_e = inputs["<n_e>"]
        n_ion_frac = inputs["ni/ne"]
        T_e = (p_e * kilo) / (n_e * kilo * eV)
        T_i = (p_i * kilo) / (n_e * n_ion_frac * kilo * eV)
        outputs["<T_e>"] = T_e
        outputs["<T_i>"] = T_i

    def setup_partials(self):
        self.declare_partials("<T_e>", ["<p_e>", "<n_e>"])
        self.declare_partials("<T_i>", ["<p_i>", "<n_e>", "ni/ne"])

    def compute_partials(self, inputs, J):
        p_e = inputs["<p_e>"]
        p_i = inputs["<p_i>"]
        n_e = inputs["<n_e>"]
        n_ion_frac = inputs["ni/ne"]

        J["<T_e>", "<p_e>"] = 1 / (n_e * eV)
        J["<T_e>", "<n_e>"] = -p_e / (eV * n_e**2)

        J["<T_i>", "<p_i>"] = 1 / (eV * n_e * n_ion_frac)
        J["<T_i>", "<n_e>"] = -p_i / (eV * n_e**2 * n_ion_frac)
        J["<T_i>", "ni/ne"] = -p_i / (eV * n_e * n_ion_frac**2)


class ThermalVelocity(om.ExplicitComponent):
    r"""Thermal velocities for a fixed mass

    Thermal velocity can be the root mean square of the velocity in any one
    dimension

    .. math::

       v_\mathrm{th} = \sqrt(T/m)

    Or in 3D, the most probable speed

    .. math::

       v_\mathrm{mps} = \sqrt(2T/m)

    Or in 3D, the root mean square of the total velocity

    .. math::

       v_\mathrm{rms3D} = \sqrt(3 T / m)

    Or the mean of the magnitude of the velocity

    .. math::

       v_\mathrm{meanmag} = \sqrt{8 T / \pi m}


    Options
    -------
    mass : float
        kg

    Inputs
    ------
    T : float
        eV, temperature

    Outputs
    -------
    vth : float
        m/s, simple thermal velocity
    v_mps : float
        m/s, Most probable speed in 3D
    v_rms : float
        m/s, Root mean square of the total velocity in 3D
    v_meanmag : float
        m/s, Mean of the magnitude of velocity

    Notes
    -----
    Units of eV are used here for temperature rather than J to avoid
    small numbers in J; this is bad for the derivatives.
    """
    def initialize(self):
        self.options.declare('mass', default=electron_mass)

    def setup(self):
        self.add_input("T", units='eV')
        self.add_output("v_th", units='m/s')
        self.add_output("v_mps", units='m/s')
        self.add_output("v_rms", units='m/s')
        self.add_output("v_meanmag", units='m/s')

    def compute(self, inputs, outputs):
        mass = self.options['mass']
        TeV = eV * inputs["T"]
        outputs["v_th"] = (TeV / mass)**(1 / 2)
        outputs["v_mps"] = (2 * TeV / mass)**(1 / 2)
        outputs["v_rms"] = (3 * TeV / mass)**(1 / 2)
        outputs["v_meanmag"] = (8 * TeV / mass / pi)**(1 / 2)

    def setup_partials(self):
        self.declare_partials("v_th", ["T"])
        self.declare_partials("v_mps", ["T"])
        self.declare_partials("v_rms", ["T"])
        self.declare_partials("v_meanmag", ["T"])

    def compute_partials(self, inputs, J):
        mass = self.options['mass']
        T = inputs["T"]
        J["v_th", "T"] = (1 * eV)**(1/2) / (2 * (T * mass)**(1 / 2))
        J["v_mps", "T"] = (2 * eV)**(1/2) / (2 * (T * mass)**(1 / 2))
        J["v_rms", "T"] = (3 * eV)**(1/2) / (2 * (T * mass)**(1 / 2))
        J["v_meanmag", "T"] = (8 * eV / pi)**(1/2) / (2 * (T * mass)**(1 / 2))


class ZeroDThermalFusionPower(om.ExplicitComponent):
    def setup(self):
        self.add_input("V", units="m**3", desc="Plasma volume")
        self.add_input("P_fus/V", units="MW/m**3")
        self.add_input("P_n/V", units="MW/m**3")
        self.add_input("P_α/V", units="MW/m**3")
        self.add_input("enhancement", desc="Fusion enhancement from p-peaking")
        self.add_output("P_fus", units="MW")
        self.add_output("P_n", units="MW")
        self.add_output("P_α", units="MW")

    def compute(self, inputs, outputs):
        f_enh = inputs["enhancement"]
        V = inputs["V"]
        outputs["P_fus"] = f_enh * V * inputs["P_fus/V"]
        outputs["P_α"] = f_enh * V * inputs["P_α/V"]
        outputs["P_n"] = f_enh * V * inputs["P_n/V"]

    def setup_partials(self):
        self.declare_partials("P_fus", ["V", "enhancement", "P_fus/V"])
        self.declare_partials("P_α", ["V", "enhancement", "P_α/V"])
        self.declare_partials("P_n", ["V", "enhancement", "P_n/V"])

    def compute_partials(self, inputs, J):
        f_enh = inputs["enhancement"]
        V = inputs["V"]
        P_fus_V = inputs["P_fus/V"]
        P_α_V = inputs["P_α/V"]
        P_n_V = inputs["P_n/V"]
        J["P_fus", "V"] = f_enh * P_fus_V
        J["P_α", "V"] = f_enh * P_α_V
        J["P_n", "V"] = f_enh * P_n_V
        J["P_fus", "enhancement"] = V * P_fus_V
        J["P_α", "enhancement"] = V * P_α_V
        J["P_n", "enhancement"] = V * P_n_V
        J["P_fus", "P_fus/V"] = f_enh * V
        J["P_α", "P_α/V"] = f_enh * V
        J["P_n", "P_n/V"] = f_enh * V


class ZeroDThermalFusion(om.Group):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        self.add_subsystem('ratecoeff',
                           SimpleRateCoeff(),
                           promotes_inputs=["T"],
                           promotes_outputs=["<σv>"])
        self.add_subsystem('rate',
                           VolumetricThermalFusionRate(),
                           promotes_inputs=["n_D", "n_T", "<σv>"],
                           promotes_outputs=[
                               "P_fus/V",
                               "P_α/V",
                               "P_n/V",
                           ])
        self.add_subsystem('power',
                           ZeroDThermalFusionPower(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])


class ZeroDPlasma(om.Group):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']
        self.add_subsystem('mix', MainIonMix(), promotes_outputs=["*"])
        self.add_subsystem('props',
                           ZeroDPlasmaProperties(config=config),
                           promotes_outputs=["*"])
        self.add_subsystem('densities',
                           ZeroDPlasmaDensities(),
                           promotes_inputs=["*", ("n_e", "<n_e>")],
                           promotes_outputs=["*"])
        self.add_subsystem('storedenergy',
                           ZeroDPlasmaStoredEnergy(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])
        self.add_subsystem('pressures',
                           ZeroDPlasmaPressures(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])
        self.add_subsystem('temperatures',
                           ZeroDPlasmaTemperatures(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])
        self.add_subsystem('vthe',
                           ThermalVelocity(mass=electron_mass),
                           promotes_inputs=[("T", "<T_e>")],
                           promotes_outputs=[('v_th', 'vth_e')])
        self.add_subsystem('th_fus',
                           ZeroDThermalFusion(),
                           promotes_inputs=[
                               ("enhancement",
                                "P_fus enhancement from p-peaking"), "n_D",
                               "n_T", "V", ("T", "<T_i>")
                           ],
                           promotes_outputs=[("P_fus", "P_fus_th"),
                                             ("P_α", "P_α_th"),
                                             ("P_n", "P_n_th")])


if __name__ == "__main__":
    uc = UserConfigurator()
    prob = om.Problem()
    prob.model = ZeroDPlasma(config=uc)
    prob.setup()
    prob.set_val("V", 455.85, units="m**3")
    prob.set_val("τ_th", 2.46, units="s")
    prob.set_val("P_loss", 83.34, units="MW")
    prob.set_val("W_fast_NBI", 11.94, units="MJ")
    prob.set_val("W_fast_α", 13.05, units="MJ")
    prob.set_val("<n_e>", 1.06e20, units="m**-3")
    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
