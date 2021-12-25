from faroes.configurator import UserConfigurator, Accessor
from faroes.plasmaformulary import AverageIonMass
from faroes.trappedparticles import TrappedParticleFractionUpperEst
import faroes.units  # noqa: F401

import openmdao.api as om

from scipy.special import hyp2f1
from scipy.constants import eV, pi, kilo, mega
from scipy.constants import physical_constants

import numpy as np

electron_mass_in_u = physical_constants["electron mass in u"][0]


class CurrentDriveProperties(om.Group):
    """Helper class to load properties
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["h_cd", "NBI", "current drive estimate"])
        acc.set_output(ivc, f, "ε fraction")
        self.add_subsystem("ivc", ivc, promotes=["*"])


class CurrentDriveBeta1(om.ExplicitComponent):
    r"""Current drive variable β1

    This is special case of the more general parameter
    for the fast ion distribution βn,

    .. math::

        \beta_n = m_i Z_\mathrm{eff} n (n+1) / (2 m_b)

    Only the :math:`n=1` term is needed to compute the current drive efficiency

    Inputs
    ------
    Ab : float
        u, Neutral beam particle mass
    Ai : float
        u, Average plasma ion mass
    Z_eff : float
        Effective ion charge

    Outputs
    -------
    β1 : float
        Fast ion distribution variable

    References
    ----------
    After Equation (44) of
    Start, D. F. H.; Cordey, J. G.; Jones, E. M.
    The Effect of Trapped Electrons on Beam Driven Currents
    in Toroidal Plasmas. Plasma Physics 1980, 22 (4), 303–316.
    https://doi.org/10.1088/0032-1028/22/4/002.
    """
    def setup(self):
        self.add_input("Z_eff", desc="Effective ion charge")
        self.add_input("Ab", units='u', desc="Neutral beam ion mass")
        self.add_input("Ai", units='u', desc="Averaged plasma ion mass")
        self.add_output("β1", desc="Current drive variable β₁")

    def compute(self, inputs, outputs):
        zeff = inputs["Z_eff"]
        Ab = inputs["Ab"]
        Ai = inputs["Ai"]
        outputs["β1"] = (Ai / Ab) * zeff

    def setup_partials(self):
        self.declare_partials("β1", ["Z_eff", "Ab", "Ai"])

    def compute_partials(self, inputs, J):
        zeff = inputs["Z_eff"]
        Ab = inputs["Ab"]
        Ai = inputs["Ai"]
        J["β1", "Z_eff"] = (Ai / Ab)
        J["β1", "Ai"] = (zeff / Ab)
        J["β1", "Ab"] = -(Ai * zeff / Ab**2)


class CurrentDriveA(om.ExplicitComponent):
    r"""Approximate current drive A

    Inputs
    ------
    Z_eff : float
        Effective ion charge
    vb : float
        Mm/s, Neutral beam initial velocity
    vth_e : float
        Mm/s, Electron thermal velocity

    Outputs
    -------
    A : float
        A function used in current drive calculations

    Notes
    -----
    Formula from Menard cell T128. I'm not sure what reference
    this is from.

    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        self.add_input("Z_eff", desc="Effective ion charge")
        self.add_input("vb",
                       units='Mm/s',
                       desc="Neutral beam ion initial velocity")
        self.add_input("vth_e", units='Mm/s', desc="Electron thermal velocity")
        self.add_output("A", desc="Variable A")
        self.const = 0.6

    def compute(self, inputs, outputs):
        vb = inputs["vb"]
        vthe = inputs["vth_e"]
        zeff = inputs["Z_eff"]
        A = 1 + self.const / ((1 + vb / vthe) * zeff)
        outputs["A"] = A

    def setup_partials(self):
        self.declare_partials("A", ["vb", "vth_e", "Z_eff"])

    def compute_partials(self, inputs, J):
        vb = inputs["vb"]
        vthe = inputs["vth_e"]
        zeff = inputs["Z_eff"]
        J["A", "vb"] = -self.const * vthe / ((vthe + vb)**2 * zeff)
        J["A", "vth_e"] = self.const * vb / ((vthe + vb)**2 * zeff)
        J["A", "Z_eff"] = -self.const * vthe / ((vthe + vb) * zeff**2)


class CurrentDriveAlphaCubed(om.ExplicitComponent):
    r"""Variable α³ for current drive calculations

    .. math::

        \alpha^3 = 0.75 \pi^{1/2} m_e (v_e/v_0)^3 (\sum n_i Z_i^2 / n_e m_i)

    This is implemented as

        \alpha^3 = 0.75 \pi^{1/2} A_e (v_e/v_0)^3 (\sum n_i Z_i^2 / n_e A_i)

    Where A_e is the electron mass in u

    Inputs
    ------
    v0 : float
        Mm/s, Initial velocity of beam ions
    ve : float
        Mm/s, Electron thermal velocity
    ne : float
        n20, Electron density
    ni : array
        n20, Plasma ion densities
    Ai : array
        u, Plasma ion masses
    Zi : array
        fundamental charge, Charges of ion species

    Outputs
    -------
    α³ : float
        Variable

    References
    ----------
    Equation (44) of
    Start, D. F. H.; Cordey, J. G.; Jones, E. M.
    The Effect of Trapped Electrons on Beam Driven Currents
    in Toroidal Plasmas. Plasma Physics 1980, 22 (4), 303–316.
    https://doi.org/10.1088/0032-1028/22/4/002.

    Notes
    -----
    This equation is referenced to be from a paper by Cordey and Haas

    """
    def setup(self):
        self.add_input("v0",
                       units="Mm/s",
                       desc="Neutral beam ion initial velocity")
        self.add_input("ve", units="Mm/s", desc="Electron thermal velocity")
        self.add_input("ne", units="n20", desc="Electron density")
        self.add_input('ni',
                       units='n20',
                       shape_by_conn=True,
                       desc="Ion field particle densities")
        self.add_input('Ai',
                       units='u',
                       shape_by_conn=True,
                       copy_shape='ni',
                       desc="Ion field particle atomic masses")
        self.add_input('Zi',
                       shape_by_conn=True,
                       copy_shape='ni',
                       desc="Ion field particle charges")
        tiny = 1e-6
        self.add_output('α³', lower=tiny, desc="Current drive variable α³")
        self.const = 0.75 * pi**(1 / 2) * electron_mass_in_u

    def compute(self, inputs, outputs):
        ve = inputs["ve"]
        v0 = inputs["v0"]
        ne = inputs["ne"]
        ni = inputs["ni"]
        ai = inputs["Ai"]
        zi = inputs["Zi"]
        s = np.sum(ni * zi**2 / ai) / ne
        α3 = self.const * (ve / v0)**3 * s
        outputs["α³"] = α3

    def setup_partials(self):
        self.declare_partials("α³", ["ve", "v0", "ne", "ni", "Ai", "Zi"])

    def compute_partials(self, inputs, J):
        ve = inputs["ve"]
        v0 = inputs["v0"]
        ne = inputs["ne"]
        ni = inputs["ni"]
        ai = inputs["Ai"]
        zi = inputs["Zi"]
        s = np.sum(ni * zi**2 / ai)
        ss = np.sum(ni * zi**2 / ai) / ne
        v_rat = (ve / v0)**3
        J["α³", "ve"] = self.const * (3 * ve**2 / v0**3) * ss
        J["α³", "v0"] = self.const * (-3) * (ve**3 / v0**4) * ss
        J["α³", "ne"] = -self.const * v_rat * s / ne**2
        J["α³", "ni"] = self.const * v_rat * zi**2 / (ai * ne)
        J["α³", "Zi"] = self.const * v_rat * ni * 2 * zi / (ai * ne)
        J["α³", "Ai"] = -self.const * v_rat * ni * zi**2 / (ai**2 * ne)


class CurrentDriveG(om.ExplicitComponent):
    r"""Neutral beam current drive variable G

    Inputs
    ------
    ftrap_u : float
        Fraction of trapped particles
    A : float
        A factor used in current drive calculations
    Zb : int
        Neutral beam ion charge
    Z_eff : float
        Effective ion charge

    Outputs
    -------
    G : float
        ?

    Notes
    -----
    From Menard's spreadsheet cell T129. I don't know where this formula
    originate. It's labeled as G(Z_eff, A) but there are additional inputs...
    """
    def setup(self):
        self.add_input("ftrap_u", desc="Trapped particle fraction")
        self.add_input("A", desc="Variable A")
        self.add_input("Zb", desc="Neutral beam ion charge")
        self.add_input("Z_eff", desc="Effective ion charge")
        self.add_output("G", lower=0, desc="Variable G")

    def compute(self, inputs, outputs):
        ftrap_u = inputs["ftrap_u"]
        A = inputs["A"]
        zb = inputs["Zb"]
        z_eff = inputs["Z_eff"]
        G = 1 + (ftrap_u * A - 1) * zb / z_eff
        outputs["G"] = G

    def setup_partials(self):
        self.declare_partials("G", ["ftrap_u", "A", "Zb", "Z_eff"],
                              method="cs")


class CurrentDriveEfficiencyTerm1and2(om.ExplicitComponent):
    r"""

    .. math::

        \frac{\tau_{se} v_0 Z_b G}{2 \pi R (1 + \alpha^3) E_{NBI}}

    .. math::

        1 + (3 - 2 \alpha^3 \beta_1) \delta / (1 + \alpha^3)^2

        \delta \equiv \left<T_e\right>/(2 E_\mathrm{NBI})

    Inputs
    ------
    τs : float
        s, Slowing time of beam particles on electrons
    v0 : float
        m/s, Beam ion initial velocity
    Zb : int
        Fundamental charges, beam ion charge
    G : float
        Parameter G(Z_eff, A)
    R : float
        m, Tokamak major radius
    α³ : float
        Current drive variable
    E_NBI : float
        keV, Beam ion initial energy
    β1 : float
        Current drive parameter
    <T_e> : float
        keV, Electron temperature

    Outputs
    -------
    line1 : float
        First line of Equation (45) of [1]
    line2 : float
        Second line of the equation

    References
    ----------
    .. [1] Equation (45) of
    Start, D. F. H.; Cordey, J. G.; Jones, E. M.
    The Effect of Trapped Electrons on Beam Driven Currents
    in Toroidal Plasmas. Plasma Physics 1980, 22 (4), 303–316.
    https://doi.org/10.1088/0032-1028/22/4/002.
    """
    def setup(self):
        self.add_input("τs", units="s", desc="Slowing time of beam ions on e⁻")
        self.add_input("v0", units="Mm/s", desc="Beam ion initial velocity")
        self.add_input("Zb", desc="Beam ion charge")
        self.add_input("G", desc="Current drive variable G")
        self.add_input("R", units="m", desc="Major radius")
        self.add_input("α³", desc="Current drive variable α³")
        self.add_input("E_NBI", units="keV", desc="Beam ion initial energy")
        self.add_input("β1", desc="Current drive variable β₁")
        self.add_input("<T_e>", units="keV", desc="Electron temperature")
        self.add_output("line1",
                        units="A/W",
                        desc="Line 1 of NBIcd eff. equation")
        self.add_output("line2", desc="Line 2 of NBIcd eff. equation")

    def compute(self, inputs, outputs):
        τs = inputs["τs"]
        v0 = inputs["v0"]
        zb = inputs["Zb"]
        G = inputs["G"]
        R = inputs["R"]
        α3 = inputs["α³"]
        E_NBI = inputs["E_NBI"]
        line1 = τs * (mega * v0) * zb * G / (2 * pi * R * (1 + α3) *
                                             (kilo * E_NBI))
        outputs["line1"] = line1

        β1 = inputs["β1"]
        Te = inputs["<T_e>"]
        line2 = 1 + (3 - 2 * α3 * β1) * Te / (2 * E_NBI * (1 + α3)**2)
        outputs["line2"] = line2

    def setup_partials(self):
        self.declare_partials("line1",
                              ["τs", "v0", "Zb", "G", "R", "α³", "E_NBI"])
        self.declare_partials("line2", ["α³", "β1", "<T_e>", "E_NBI"])

    def compute_partials(self, inputs, J):
        τs = inputs["τs"]
        v0 = inputs["v0"]
        zb = inputs["Zb"]
        G = inputs["G"]
        R = inputs["R"]
        α3 = inputs["α³"]
        E_NBI = inputs["E_NBI"]
        numer = τs * (mega * v0) * zb * G
        denom = (2 * pi * R * (1 + α3) * (kilo * E_NBI))
        J["line1", "τs"] = (mega * v0) * zb * G / denom
        J["line1", "v0"] = τs * mega * zb * G / denom
        J["line1", "Zb"] = τs * (mega * v0) * G / denom
        J["line1", "G"] = τs * (mega * v0) * zb / denom
        J["line1", "R"] = -numer / denom / R
        J["line1", "E_NBI"] = -numer / denom / E_NBI
        J["line1", "α³"] = -numer / (2 * pi * R * (1 + α3)**2 * (kilo * E_NBI))

        β1 = inputs["β1"]
        Te = inputs["<T_e>"]
        J["line2", "α³"] = (-3 + (α3 - 1) * β1) * Te / ((1 + α3)**3 * E_NBI)
        J["line2", "β1"] = -α3 * Te / ((1 + α3)**2 * E_NBI)
        J["line2", "<T_e>"] = (3 - 2 * α3 * β1) / (2 * (1 + α3)**2 * E_NBI)
        J["line2",
          "E_NBI"] = (-3 + 2 * α3 * β1) * Te / (2 * (1 + α3)**2 * E_NBI**2)


class CurrentDriveEfficiencyTerm3(om.ExplicitComponent):
    r"""Integral for current drive efficiency calculations

    .. math::

        i = \int_0^1 x^{3 + \beta_1}
           \left(\frac{1 + \alpha^3}{x^3 + \alpha^3}\right)^{1+\beta_1/3} \; dx

    Inputs
    ------
    α³ : float
        A parameter used in current drive calculations
    β1 : float
        A parameter used in current drive calculations

    Outputs
    -------
    line3 : float
        The integral

    References
    ----------
    Line 3 of Equation (45) of
    Start, D. F. H.; Cordey, J. G.; Jones, E. M.
    The Effect of Trapped Electrons on Beam Driven Currents
    in Toroidal Plasmas. Plasma Physics 1980, 22 (4), 303–316.
    https://doi.org/10.1088/0032-1028/22/4/002.

    Notes
    -----
    There is no general analytic formula for derivatives of the 2F1 function
    with respect to the parameters, so finite differencing is used here.
    The function is fairly smooth so it should be acceptable.

    """
    def setup(self):
        self.add_input("α³", desc="Current drive variable α³")
        self.add_input("β1", desc="Current drive variable β₁")
        self.add_output("line3", lower=0, desc="Line 3 of NBIcd eff. equation")

    def compute(self, inputs, outputs):
        β1 = inputs["β1"]
        α3 = inputs["α³"]

        if α3 < 1e-10:
            raise om.AnalysisError("Current drive α3 is too small")

        p1 = (3 + β1) / 3
        p2 = (4 + β1) / 3
        p3 = (7 + β1) / 3
        arg = -1 / α3
        hyp = hyp2f1(p1, p2, p3, arg)
        result = (α3 / (1 + α3))**(-1 - β1 / 3) * hyp / (4 + β1)
        outputs["line3"] = result

    def setup_partials(self):
        self.declare_partials("line3", ["β1"], method="fd")
        self.declare_partials("line3", ["α³"])

    def compute_partials(self, inputs, J):
        β1 = inputs["β1"]
        α3 = inputs["α³"]
        p3 = (7 + β1) / 3
        arg = -1 / α3
        hyp = hyp2f1(1, 4 / 3, p3, arg)
        exp = (-β1 / 3)
        prefactor = (1 + 1 / α3)**exp * (α3 / (1 + α3))**exp / (3 * α3**2 *
                                                                (4 + β1))
        result = prefactor * (-α3 * (4 + β1) + (1 + α3 * (4 + β1)) * hyp)
        J["line3", "α³"] = result


class CurrentDriveEfficiencyEquation(om.ExplicitComponent):
    r"""Current drive efficiency

    Inputs
    ------
    line1 : float
        A/W
    line2 : float
        Second line of the equation
    line3 : float
        Third line of the equation

    Outputs
    -------
    It/P : float
        A/W, Ratio of the net current flowing parallel to the magnetic field
            to the injected neutral beam power

    References
    ----------
    Line 3 of Equation (45) of
    Start, D. F. H.; Cordey, J. G.; Jones, E. M.
    The Effect of Trapped Electrons on Beam Driven Currents
    in Toroidal Plasmas. Plasma Physics 1980, 22 (4), 303–316.
    https://doi.org/10.1088/0032-1028/22/4/002.
    """
    def setup(self):
        self.add_input("line1",
                       units="A/W",
                       desc="Line 1 of NBIcd eff.  equation")
        self.add_input("line2", desc="Line 2 of NBIcd eff. equation")
        self.add_input("line3", desc="Line 3 of NBIcd eff. equation")
        self.add_output("It/P",
                        units="A/W",
                        desc="NBI current drive 'efficiency'")

    def compute(self, inputs, outputs):
        line1 = inputs["line1"]
        line2 = inputs["line2"]
        line3 = inputs["line3"]
        outputs["It/P"] = line1 * line2 * line3

    def setup_partials(self):
        self.declare_partials("It/P", ["line1", "line2", "line3"])

    def compute_partials(self, inputs, J):
        line1 = inputs["line1"]
        line2 = inputs["line2"]
        line3 = inputs["line3"]
        J["It/P", "line1"] = line2 * line3
        J["It/P", "line2"] = line1 * line3
        J["It/P", "line3"] = line1 * line2


class CurrentDriveEfficiency(om.Group):
    r"""

    Inputs
    ------
    Ab : float
        u, mass of beam ions
    Zb : int
        Fundamental charges, charge of beam ions
    vb : float
        m/s, Initial velocity of beam ions
    Eb : float
        keV, Initial energy of beam ions

    R0 : float
        m, Major radius of plasma
    A : float
        Aspect ratio R/a

    Z_eff : float
        Effective ion charge
    ne : float
        n20, plasma electron density
    <T_e> : float
        keV, Average plasma electron temperature
    vth_e : float
        m/s, Electron thermal velocity

    τs : float
        s, Slowing time of beam ions on electrons

    ni : array
        n20, densities of plasma ions
    Ai : array
        u, Masses of plasma ions
    Zi : array
        Fundamental charges, charges of plasma ions

    Outputs
    -------
    ftrap_u : float
        Trapped particle fraction
    It/P : float
        A/W, Ratio of the net current flowing parallel to the magnetic field
            to the injected neutral beam power
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem('props',
                           CurrentDriveProperties(config=config),
                           promotes_outputs=["ε fraction"])
        self.add_subsystem(
            'eps_neo',
            om.ExecComp(
                "eps_neo = eps_frac * eps",
                eps_frac={'desc': "Generalized minor radius for evaluation"},
                eps={'desc': "Inverse aspect ratio"},
                eps_neo={'desc': "Inverse aspect ratio for evaluation"},
            ),
            promotes_inputs=[("eps_frac", "ε fraction"), ("eps", "ε")],
            promotes_outputs=[("eps_neo", "ε_neoclass")])
        self.add_subsystem('A_bar',
                           AverageIonMass(),
                           promotes_inputs=["ni", "Ai"],
                           promotes_outputs=["A_bar"])
        self.add_subsystem('beta1',
                           CurrentDriveBeta1(),
                           promotes_inputs=["Z_eff", "Ab", ("Ai", "A_bar")],
                           promotes_outputs=["β1"])
        self.add_subsystem('A',
                           CurrentDriveA(),
                           promotes_inputs=["Z_eff", "vb", "vth_e"],
                           promotes_outputs=["A"])
        self.add_subsystem('ftrapped',
                           TrappedParticleFractionUpperEst(),
                           promotes_inputs=[("ε", "ε_neoclass")],
                           promotes_outputs=["ftrap_u"])
        self.add_subsystem('alphacubed',
                           CurrentDriveAlphaCubed(),
                           promotes_inputs=[("v0", "vb"), ("ve", "vth_e"),
                                            "ne", "ni", "Ai", "Zi"],
                           promotes_outputs=["α³"])
        self.add_subsystem('G',
                           CurrentDriveG(),
                           promotes_inputs=["ftrap_u", "A", "Zb", "Z_eff"],
                           promotes_outputs=["G"])
        self.add_subsystem("line1and2",
                           CurrentDriveEfficiencyTerm1and2(),
                           promotes_inputs=[
                               "τs",
                               ("v0", "vb"),
                               "Zb",
                               "G",
                               ("R", "R0"),
                               "α³",
                               ("E_NBI", "Eb"),
                               "β1",
                               "<T_e>",
                           ],
                           promotes_outputs=["line1", "line2"])
        self.add_subsystem("line3",
                           CurrentDriveEfficiencyTerm3(),
                           promotes_inputs=["α³", "β1"],
                           promotes_outputs=["line3"])
        self.add_subsystem('eff',
                           CurrentDriveEfficiencyEquation(),
                           promotes_inputs=["*"],
                           promotes_outputs=["It/P"])


class NBICurrent(om.ExplicitComponent):
    r"""Incorporate beams with multiple energy components

    .. math::

        I_\mathrm{NBI} = fudge * \sum (S Eb It/P)

    Inputs
    ------
    S : array
        1/s, Neutral beam source rates
    It/P: array
        A/W, efficiency of current drive for each source
    Eb : array
        keV, Initial energy of beam ions for each source
    fudge : float
        Overall fudge factor.
        This will likely be a constant during a given run,
        but can be used to match other calculations

    Outputs
    -------
    I_NBI : float
        MA, total neutral-beam-driven current
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"]
        self.add_input("S",
                       units="1/s",
                       shape_by_conn=True,
                       desc="Neutral beam source rate")
        self.add_input("Eb",
                       units="keV",
                       shape_by_conn=True,
                       copy_shape="S",
                       desc="Neutral beam initial energy")
        self.add_input("It/P",
                       units="A/W",
                       shape_by_conn=True,
                       copy_shape="S",
                       desc="Current drive efficiency")
        self.add_output("I_NBI",
                        units="MA",
                        desc="Neutral-beam-driven current")

        if config is not None:
            self.config = self.options["config"]
            ac = self.config.accessor(["h_cd", "NBI"])
            c = ac(["current drive estimate", "fudge factor"])
            self.add_input("fudge", val=c)
        else:
            self.add_input("fudge", val=1.0)

    def compute(self, inputs, outputs):
        S = inputs["S"]
        Eb = inputs["Eb"]
        eff = inputs["It/P"]
        f = inputs["fudge"]
        outputs["I_NBI"] = f * kilo * eV * np.sum(S * Eb * eff) / mega

    def setup_partials(self):
        self.declare_partials("I_NBI", ["S", "Eb", "It/P"])
        self.declare_partials("I_NBI", ["fudge"])

    def compute_partials(self, inputs, J):
        S = inputs["S"]
        Eb = inputs["Eb"]
        eff = inputs["It/P"]
        f = inputs["fudge"]
        J["I_NBI", "S"] = f * kilo * eV * Eb * eff / mega
        J["I_NBI", "Eb"] = f * kilo * eV * S * eff / mega
        J["I_NBI", "It/P"] = f * kilo * eV * Eb * S / mega
        J["I_NBI", "fudge"] = f * kilo * eV * np.sum(S * Eb * eff) / mega


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model.add_subsystem("ivc",
                             om.IndepVarComp("ni", val=np.ones(3),
                                             units="n20"),
                             promotes_outputs=["*"])
    prob.model.add_subsystem("cde",
                             CurrentDriveEfficiency(config=uc),
                             promotes_inputs=["*"])

    prob.setup()

    prob.set_val("Ab", 2, units="u")
    prob.set_val("Zb", 1)
    prob.set_val("vb", 6922, units="km/s")
    prob.set_val("Eb", 500, units="keV")

    prob.set_val("R0", 3.0, units="m")
    prob.set_val("ε", 1 / 1.6)

    prob.set_val("Z_eff", 2)
    prob.set_val("ne", 1.06, units="n20")
    prob.set_val("<T_e>", 9.20, units="keV")
    prob.set_val("vth_e", 56922, units="km/s")

    prob.set_val("τs", 0.599, units="s")

    prob.set_val("ni", np.array([0.424, 0.424, 0.0353]), units="n20")
    prob.set_val("Ai", [2, 3, 12], units="u")
    prob.set_val("Zi", [1, 1, 6])

    prob.run_driver()

    prob.model.list_inputs(val=True, print_arrays=True, desc=True, units=True)
    prob.model.list_outputs(val=True, print_arrays=True, desc=True, units=True)
