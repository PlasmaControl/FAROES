from faroes.configurator import UserConfigurator, Accessor
from faroes.util import tube_segment_volume
from faroes.util import tube_segment_volume_derivatives

import openmdao.api as om

from scipy.constants import mu_0, pi
from scipy.constants import kilo, mega, giga
from scipy.special import ellipk, ellipe
import numpy as np


class CentralSolenoidProperties(om.Group):
    """Helper class to load properties
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["materials", "CS winding pack"])
        acc.set_output(ivc, f, "j_max", units='MA/m**2')
        acc.set_output(ivc,
                       f,
                       "B_max",
                       component_name='B_max_conductor',
                       units='T')
        self.add_subsystem("ivc", ivc, promotes=["*"])


class ThinSolenoidInductance(om.ExplicitComponent):
    r"""Modified-inductance of a thin-wall solenoid

    Here, the quantity of modified-inductance is defined such that

    .. math::

        W_b = (1/2) L' j_s^2

    where the surface current js has units of A/m.

    Inputs
    ------
    r : float
        m, Aspect ratio
    h : float
        m, length (height) of the solenoid

    Outputs
    -------
    L_line : float
        H m**2, inductance

    Reference
    ---------
    H. Nagaoka, "The Inductance Coefficients of Solenoids,"
    Journal of the College of Science, Imperial University, Tokyo,
    Vol. 27, Article 6, 1909, p. 33.
    """
    def setup(self):
        self.add_input("r", units="m", desc="radius")
        self.add_input("h", units="m", desc="length")

        L_line_ref = 1e-4
        self.add_output(
            "L_line",
            units="H*m**2",
            ref=L_line_ref,
            desc="Modified inductance, to use with current-per-meter")

    def L_infinite(self, r, h):
        """Modified-inductance of an infinite solenoid

        Parameters
        ----------
        r : float
            m, radius
        h : float
            m, length of the solenoid
        """
        a = self.area(r)
        return mu_0 * a * h

    def area(self, r):
        return pi * r**2

    def k_L(self, r, h):
        """Nagaoka coefficient
        """
        k = np.sqrt(4 * r**2 / (4 * r**2 + h**2))
        kp = np.sqrt(1 - k**2)
        f1 = (4 / (3 * pi * kp))
        f2 = (kp / k)**2 * (ellipk(k**2) - ellipe(k**2)) + ellipe(k**2) - k
        return f1 * f2

    def compute(self, inputs, outputs):
        r = inputs["r"]
        h = inputs["h"]
        L_inf = self.L_infinite(r, h)
        k_L = self.k_L(r, h)

        outputs["L_line"] = k_L * L_inf

    def setup_partials(self):
        self.declare_partials('L_line', ['r', 'h'])

    def compute_partials(self, inputs, J):
        r = inputs['r']
        h = inputs['h']

        k_sq = 4 * r**2 / (4 * r**2 + h**2)

        prefactor = 4 * r * mu_0
        term_1 = -2 * r
        term_2 = np.sqrt(h**2 + 4 * r**2) * ellipe(k_sq)
        J["L_line", "r"] = prefactor * (term_1 + term_2)

        prefactor = h * mu_0 * np.sqrt(h**2 + 4 * r**2)
        term_1 = -ellipe(k_sq)
        term_2 = ellipk(k_sq)
        J["L_line", "h"] = prefactor * (term_1 + term_2)


class FiniteBuildCentralSolenoid(om.ExplicitComponent):
    r""" Finite-thickness solenoid

    Calculates the central B assuming an infinite length

    Inputs
    ------
    j : float
        MA/m**2, smeared solenoid current density
    j_max : float
        MA/m**2, maximum smeared solenoid current density
    R_in : float
        m, solenoid inner radius
    R_out : float
        m, solenoid outer radius
    h : float
        m, height
    B_max_conductor : float
        T, maximum B on conductor

    """
    def setup(self):
        self.add_input("R_out", units='m', desc="outer casing radius")
        self.add_input("R_in", units='m', desc="inner casing radius")
        self.add_input("B_max_conductor",
                       units='T',
                       desc="Maximum field on conductor")
        self.add_input("h", units='m', desc="height")
        self.add_input("j", units='MA/m**2', desc="smeared current density")
        self.add_input("j_max",
                       units='MA/m**2',
                       desc="smeared current density limit")

        self.add_output("B_cond_constraint",
                        desc="Remaining fraction of maximum B on conductor")
        self.add_output("j_constraint",
                        desc="Remaining fraction of maximum current density")
        self.add_output("A_eff",
                        units="m**2",
                        lower=0,
                        desc="Effective solenoid area")
        self.add_output("r_eff",
                        units="m",
                        lower=0,
                        desc="Effective solenoid radius")
        self.add_output("B", units="T", lower=0, desc="Maximum field")
        Φ_ref = 10
        self.add_output("Φ_single",
                        units="Wb",
                        ref=Φ_ref,
                        lower=0,
                        desc="Single-swing flux")
        self.add_output("Φ_double",
                        units="Wb",
                        ref=Φ_ref,
                        lower=0,
                        desc="Double-swing flux")

        self.add_output("V", units="m**3", desc="volume")
        conductor_ref = 1e6
        self.add_output("conductor_quantity",
                        units="kA * m",
                        lower=0,
                        ref=conductor_ref,
                        desc="volume")

    def b_bore(self, j, r_inner, r_outer):
        """Central maximum B field.

        Assumes an infinite solenoid.
        """
        B = mu_0 * j * (r_outer - r_inner)
        return B

    def area_effective(self, r_i, r_o):
        """Effective area of a constant-current-density solenoid

        Returns
        -------
        A_eff : float
            m**2
        """
        return pi * (r_i**2 + r_o**2 + r_i * r_o) / 3

    def conductor_kAm(self, r_i, r_o, h, j):
        """Length of conductor in kA*m

        Parameters
        ----------
        r_i : float
            m, inner radius
        r_o : float
            m, outer radius
        h : float
            m, height
        j : float
            A/m**2

        Returns
        -------
        conductor length, kA*m
        """
        return j * tube_segment_volume(r_i, r_o, h) / kilo

    def one_way_flux(self, r_i, r_o, B):
        """Stored flux
        Note: assumes an infinite-length solenoid

        Returns: Wb
        """
        area_effective = self.area_effective(r_i, r_o)
        Phi = B * area_effective
        return Phi

    def double_swing_flux(self, r_i, r_o, B):
        """Twice the one-way flux

        Returns: Phi [Wb]
        """
        Phi = 2 * self.one_way_flux(r_i, r_o, B)
        return Phi

    def compute(self, inputs, outputs):
        j = inputs["j"] * mega
        j_max = inputs["j_max"] * mega
        outputs["j_constraint"] = (j_max - j) / j_max

        r_i = inputs["R_in"]
        r_o = inputs["R_out"]
        h = inputs["h"]

        a_eff = self.area_effective(r_i, r_o)
        outputs["A_eff"] = a_eff
        outputs["r_eff"] = np.sqrt(a_eff / pi)

        B = self.b_bore(j, r_i, r_o)
        outputs["B"] = B

        B_max_cond = inputs["B_max_conductor"]
        outputs["B_cond_constraint"] = (B_max_cond - B) / B_max_cond

        Φ_single = self.one_way_flux(r_i, r_o, B)
        Φ_double = self.double_swing_flux(r_i, r_o, B)
        outputs["Φ_single"] = Φ_single
        outputs["Φ_double"] = Φ_double

        outputs["V"] = tube_segment_volume(r_i, r_o, h)
        outputs["conductor_quantity"] = self.conductor_kAm(r_i, r_o, h, j)

    def setup_partials(self):
        self.declare_partials('j_constraint', ['j', 'j_max'])
        self.declare_partials('B', ['j', 'R_in', 'R_out'])
        self.declare_partials('B_cond_constraint',
                              ['j', 'R_in', 'R_out', 'B_max_conductor'])
        self.declare_partials('A_eff', ['R_in', 'R_out'])
        self.declare_partials('r_eff', ['R_in', 'R_out'])
        self.declare_partials('Φ_single', ['j', 'R_in', 'R_out'])
        self.declare_partials('Φ_double', ['j', 'R_in', 'R_out'])
        self.declare_partials('V', ['h', 'R_in', 'R_out'])
        self.declare_partials('conductor_quantity',
                              ['h', 'j', 'R_in', 'R_out'])

    def compute_partials(self, inputs, J):
        """Need to be careful since the j's are in MA/m**2
        """
        j = inputs["j"] * mega
        j_max = inputs["j_max"] * mega
        r_i = inputs["R_in"]
        r_o = inputs["R_out"]
        h = inputs["h"]
        J['j_constraint', 'j'] = -mega / j_max
        J['j_constraint', 'j_max'] = mega * j / j_max**2

        J['B', 'j'] = mu_0 * (r_o - r_i) * mega
        J['B', 'R_in'] = -j * mu_0
        J['B', 'R_out'] = j * mu_0

        B_max = inputs["B_max_conductor"]
        J['B_cond_constraint', 'j'] = -(r_o - r_i) * mu_0 * mega / B_max
        J['B_cond_constraint', 'R_in'] = j * mu_0 / B_max
        J['B_cond_constraint', 'R_out'] = -j * mu_0 / B_max
        J['B_cond_constraint',
          'B_max_conductor'] = j * mu_0 * (r_o - r_i) / B_max**2

        J['A_eff', 'R_in'] = pi * (2 * r_i + r_o) / 3
        J['A_eff', 'R_out'] = pi * (r_i + 2 * r_o) / 3

        denom = 2 * np.sqrt(3 * (r_o**2 + r_i**2 + r_o * r_i))
        J['r_eff', 'R_in'] = (2 * r_i + r_o) / denom
        J['r_eff', 'R_out'] = (r_i + 2 * r_o) / denom

        J['Φ_single', 'j'] = mu_0 * mega * pi * (r_o**3 - r_i**3) / 3
        J['Φ_double', 'j'] = 2 * J['Φ_single', 'j']

        J['Φ_single', 'R_in'] = -j * mu_0 * pi * r_i**2
        J['Φ_double', 'R_in'] = 2 * J['Φ_single', 'R_in']

        J['Φ_single', 'R_out'] = j * mu_0 * pi * r_o**2
        J['Φ_double', 'R_out'] = 2 * J['Φ_single', 'R_out']

        V = tube_segment_volume(r_i, r_o, h)
        dV = tube_segment_volume_derivatives(r_i, r_o, h)
        J['V', 'R_in'] = dV['r_i']
        J['V', 'R_out'] = dV['r_o']
        J['V', 'h'] = dV['h']
        J['conductor_quantity', 'R_in'] = j * J['V', 'R_in'] / kilo
        J['conductor_quantity', 'R_out'] = j * J['V', 'R_out'] / kilo
        J['conductor_quantity', 'h'] = j * J['V', 'h'] / kilo
        J['conductor_quantity', 'j'] = mega * V / kilo


class ThinSolenoidStoredEnergy(om.ExplicitComponent):
    r"""Stored energy of a thin-walled solenoid
    """
    def setup(self):
        self.add_input("jl",
                       units='MA/m',
                       desc="smeared linear current density")
        self.add_input("L_line", units="H*m**2", desc="Modified inductance")
        self.add_output("W_b", units="GJ", desc="Stored energy")

    def compute(self, inputs, outputs):
        jl = inputs["jl"] * mega
        L_line = inputs["L_line"]
        W_b = (1 / 2) * L_line * jl**2 / giga
        outputs["W_b"] = W_b

    def setup_partials(self):
        self.declare_partials('W_b', ['jl', 'L_line'])

    def compute_partials(self, inputs, J):
        jl = inputs["jl"] * mega
        L_line = inputs["L_line"]
        J['W_b', 'jl'] = mega * L_line * jl / giga
        J['W_b', 'L_line'] = (1 / 2) * jl**2 / giga


class FiniteSolenoidStresses(om.ExplicitComponent):
    r"""Module to compute stresses

    Inputs
    ------
    j : float
        A/m**2, smeared solenoid current density
    R_in : float
        m, solenoid inner radius
    R_out : float
        m, solenoid outer radius
    B : float
        T, central solenoid max B at R_in
    p_b : float
        Pa, bucking pressure

    Outputs
    -------
    Inner σ_θ : float
        Pa, total tension on the innermost conductors.
        Negative values mean compression.

    Notes
    -----
    It's nice to have all the CS in compression.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        self.add_input("j", units='MA/m**2', desc="smeared current density")
        self.add_input("R_out", units='m', desc="outer casing radius")
        self.add_input("R_in", units='m', desc="inner casing radius")
        self.add_input("B", units="T", desc="Idealized central field")
        self.add_input("p_b", units="MPa", desc="Bucking pressure")

        self.add_output("Inner σ_θ",
                        units="MPa",
                        desc="Stress on innermost turns at max field")

        self.add_output("Inner σ_θ no current",
                        units="MPa",
                        desc="Stress on innermost turns with no current")

    def outer_pressure_hoop_stress(self, r_i, r_o, p_o):
        """Stress at inner wall from outer pressure

        Assumes an infinitely long tube

        Parameters
        ----------
        r_i : float
            m, inner radius of tube
        r_o : float
            m, outer radius of tube
        p_o : float
            MPa, pressure at outer wall of tube

        Reference
        ---------
        https://www.engineeringtoolbox.com/stress-thick-walled-tube-d_949.html
        """
        sigma_c = -2 * p_o * r_o**2 / (r_o**2 - r_i**2)
        return sigma_c

    def outward_hoop_stress(self, j, B, R):
        """Local outward hoop stress [MPa]

        Parameters
        ----------
        j : float
            MA/m**2, local current density
        B : float
            Tesla, local magnetic field
        R : float
            meters, radius of wire loop

        Return
        ------
        stress : MPa
        """
        return j * B * R

    def compute(self, inputs, outputs):
        j = inputs["j"]
        r_i = inputs["R_in"]
        r_o = inputs["R_out"]
        p_bucking = inputs["p_b"]
        B = inputs["B"]

        σ_c = self.outer_pressure_hoop_stress(r_i, r_o, p_bucking)
        jbr = self.outward_hoop_stress(j, B, r_i)
        outputs["Inner σ_θ"] = σ_c + jbr
        outputs["Inner σ_θ no current"] = σ_c

    def setup_partials(self):
        self.declare_partials('Inner σ_θ', ['j', 'R_in', 'R_out', 'p_b', 'B'])
        self.declare_partials('Inner σ_θ no current', ['R_in', 'R_out', 'p_b'])

    def compute_partials(self, inputs, J):
        j = inputs["j"]
        r_i = inputs["R_in"]
        r_o = inputs["R_out"]
        B = inputs["B"]
        p_b = inputs["p_b"]

        J["Inner σ_θ", "j"] = B * r_i
        J["Inner σ_θ", "B"] = j * r_i
        denom = (r_o**2 - r_i**2)
        J["Inner σ_θ", "R_in"] = B * j - 4 * p_b * r_i * r_o**2 / denom**2
        J["Inner σ_θ", "R_out"] = 4 * p_b * r_i**2 * r_o / denom**2
        J["Inner σ_θ", "p_b"] = -2 * r_o**2 / denom

        denom = (r_o**2 - r_i**2)
        J["Inner σ_θ no current", "R_in"] = -4 * p_b * r_i * r_o**2 / denom**2
        J["Inner σ_θ no current", "R_out"] = 4 * p_b * r_i**2 * r_o / denom**2
        J["Inner σ_θ no current", "p_b"] = -2 * r_o**2 / denom


class CentralSolenoid(om.Group):
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        config = self.options['config']

        self.add_subsystem("props",
                           CentralSolenoidProperties(config=config),
                           promotes_outputs=['*'])
        self.add_subsystem('solenoid',
                           FiniteBuildCentralSolenoid(),
                           promotes_inputs=[
                               "B_max_conductor", "j_max", "R_in", "R_out",
                               "j", "h"
                           ],
                           promotes_outputs=["B_cond_constraint", "B"])
        self.add_subsystem('inductance',
                           ThinSolenoidInductance(),
                           promotes_inputs=['h'],
                           promotes_outputs=['L_line'])
        self.add_subsystem('stored_energy',
                           ThinSolenoidStoredEnergy(),
                           promotes_inputs=['jl', 'L_line'],
                           promotes_outputs=['W_b'])
        self.add_subsystem('stresses',
                           FiniteSolenoidStresses(),
                           promotes_inputs=["R_in", "R_out", "j", "B", "p_b"])

        self.add_subsystem('connector_jl',
                           om.ExecComp('jl = j * (R_out - R_in)',
                                       jl={'units': 'MA/m'},
                                       j={'units': 'MA/m**2'},
                                       R_in={'units': 'm'},
                                       R_out={'units': 'm'}),
                           promotes_inputs=['j', 'R_in', 'R_out'])
        self.connect('solenoid.r_eff', ['inductance.r'])


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model = CentralSolenoid(config=uc)

    prob.setup()

    prob.set_val('R_in', 1.1, 'm')
    prob.set_val('R_out', 1.5, 'm')
    prob.set_val('j', 30, 'MA/m**2')
    prob.set_val('h', 5, 'm')
    prob.set_val('p_b', 250, 'MPa')

    prob.run_driver()

    prob.model.list_inputs(val=True)
    prob.model.list_outputs(val=True)
