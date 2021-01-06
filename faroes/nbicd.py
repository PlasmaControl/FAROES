import openmdao.api as om
from plasmapy.particles import Particle
from faroes.configurator import UserConfigurator, Accessor
from astropy import units as apunits
from scipy.special import hyp2f1
from scipy.constants import eV, pi
from scipy.constants import physical_constants
import faroes.units  # noqa: F401
import numpy as np

electron_mass_in_u = physical_constants["electron mass in u"][0]


class CurrentDriveA(om.ExplicitComponent):
    r"""Approximate current drive A

    Inputs
    ------
    Z_eff : float
        Effective plasma charge
    vb : float
        m/s, Neutral beam initial velocity
    vth_e : float
        m/s, Electron thermal velocity

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
        self.add_input("Z_eff")
        self.add_input("vb", units='km/s')
        self.add_input("vth_e", units='km/s')
        self.add_output("A")
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


class TrappedParticleFractionUpperEst(om.ExplicitComponent):
    r"""Upper estimate for the trapped particle fraction on a flux surface

    Notes
    -----
    This is derived for "... the case of concentric, elliptical
    flux surfaces that are adequate to describe low-β,
    up-down symmetric equilibria"

    Inputs
    ------
    ε : float
        Inverse aspect ratio of flux surface

    Outputs
    -------
    ftrap_u : float
        Upper estimate of the trapped particle fraction on that surface

    References
    ----------
    Equation (13) of
    Lin‐Liu, Y. R.; Miller, R. L. Upper and Lower Bounds
    of the Effective Trapped Particle Fraction in General Tokamak Equilibria.
    Physics of Plasmas 1995, 2 (5), 1666–1668.
    https://doi.org/10.1063/1.871315.
    """
    def setup(self):
        self.add_input("ε")
        self.add_output("ftrap_u")

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


class CurrentDriveAlphaCubed(om.ExplicitComponent):
    r"""Parameter α³ for current drive calculations

    .. math::

        \alpha^3 = 0.75 \pi^{1/2} m_e (v_e/v_0)^3 (\sum n_i Z_i^2 / n_e m_i)

    This is implemented as

        \alpha^3 = 0.75 \pi^{1/2} A_e (v_e/v_0)^3 (\sum n_i Z_i^2 / n_e A_i)

    Where A_e is the electron mass in u

    Inputs
    ------
    v0 : float
        m/s, Initial velocity of beam ions
    ve : float
        m/s, Electron thermal velocity
    ne : float
        n20, electron density
    ni : array
        n20, ion densities
    Ai : array
        u, ion masses
    Zi : array
        units of fundamental charge of ions

    Outputs
    -------
    α³ : float
        Parameter

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
        self.add_input("v0", units="Mm/s")
        self.add_input("ve", units="Mm/s")
        self.add_input("ne", units="n20")
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
        self.add_output('α³')
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


class CurrentDriveIntegral(om.ExplicitComponent):
    r"""Integral for current drive efficiency calculations

    .. math::

        i = \int_0^1 x^{3 + \beta_1} \left(\frac{1 + \alpha^3}{x^3 + \alpha^3}\right)^{1+\beta_1/3} \; dx

    Inputs
    ------
    α³ : float
        A parameter used in current drive calculations
    β1 : float
        A parameter used in current drive calculations

    Outputs
    -------
    i : float
        The integral

    References
    ----------
    Equation (45) of
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
        self.add_input("α³")
        self.add_input("β1")
        self.add_output("i")

    def compute(self, inputs, outputs):
        β1 = inputs["β1"]
        α3 = inputs["α³"]
        p1 = (3 + β1) / 3
        p2 = (4 + β1) / 3
        p3 = (7 + β1) / 3
        arg = -1 / α3
        hyp = hyp2f1(p1, p2, p3, arg)
        result = (α3 / (1 + α3))**(-1 - β1 / 3) * hyp / (4 + β1)
        outputs["i"] = result

    def setup_partials(self):
        self.declare_partials("i", ["β1"], method="fd")
        self.declare_partials("i", ["α³"])

    def compute_partials(self, inputs, J):
        β1 = inputs["β1"]
        α3 = inputs["α³"]
        p1 = (3 + β1) / 3
        p2 = (4 + β1) / 3
        p3 = (7 + β1) / 3
        arg = -1 / α3
        hyp = hyp2f1(p1, p2, p3, arg)

        exp = (-1 - β1 / 3)
        term1 = (α3 / (1 + α3))**exp * ((1 + 1 / α3)**exp - hyp) / (3 * α3)
        term2 = (α3 / (1 + α3))**(exp - 1) * (-α3 / (1 + α3)**2 + 1 /
                                              (1 + α3)) * exp * hyp / (4 + β1)
        result = -term1 + term2
        J["i", "α³"] = result


class CurrentDriveG(om.ExplicitComponent):
    r"""Neutral beam current drive variable G

    Inputs
    ------
    ftrap_u : float
        Fraction of trapped particles(?)
    A : float
        A factor used in current drive calculations
    Zb : int
        Charge of beam ions
    Z_eff : float
        Plasma effective charge

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
        self.add_input("ftrap_u")
        self.add_input("A")
        self.add_input("Zb")
        self.add_input("Z_eff")
        self.add_output("G")

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


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model = CurrentDriveA()

    prob.setup()
    prob.run_driver()

    prob.model.list_inputs(values=True)
    prob.model.list_outputs(values=True)
