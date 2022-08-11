import openmdao.api as om

import faroes.units  # noqa: F401
from scipy.constants import mu_0
import numpy as np


class AlfvenSpeed(om.ExplicitComponent):
    r"""Alfvén speed

    .. math::

       V_A = \left|B\right| / \sqrt{μ_0 ρ}

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


class AverageIonMass(om.ExplicitComponent):
    r"""Average ion mass

    .. math::

       \bar{A} = \frac{\sum_i n_i A_i }{ \sum_i n_i}

    Inputs
    ------
    ni : array
        n20, Ion densities
    Ai : array
        u, Ion atomic masses

    Outputs
    -------
    A_bar : float
        u, Average ion mass
    """
    def setup(self):
        self.add_input("ni",
                       units="n20",
                       shape_by_conn=True,
                       desc="Ion field particle densities")
        self.add_input("Ai",
                       units="u",
                       shape_by_conn=True,
                       copy_shape="ni",
                       desc="Ion field particle atomic masses")
        self.add_output("A_bar", units="u", lower=0, desc="Average ion mass")

    def compute(self, inputs, outputs):
        ni = inputs["ni"]
        Ai = inputs["Ai"]
        outputs["A_bar"] = np.sum(ni * Ai) / np.sum(ni)

    def setup_partials(self):
        self.declare_partials("A_bar", ["ni", "Ai"])

    def compute_partials(self, inputs, J):
        ni = inputs["ni"]
        Ai = inputs["Ai"]
        numer = np.sum(ni * Ai)
        denom = np.sum(ni)
        J["A_bar", "ni"] = Ai / denom - numer / denom**2
        J["A_bar", "Ai"] = ni / denom


class CoulombLogarithmElectrons(om.ExplicitComponent):
    r"""Electron coulomb logarithm

    .. math::
        \log Λ_e = 31.3 - \log(\sqrt{n_e} / T_e)

    Where ne is in per cubic meter and Te is in eV.
    The formula is from :footcite:t:`sauter_neoclassical_1999`.

    Inputs
    ------
    ne : float
       n20, electron density
    Te : float
       eV, electron temperature

    Outputs
    -------
    logΛe : float
       Electron coulomb logarithm

    Notes
    -----
    This computation may be somewhat specific for fast-particle slowing.
    Need to check.

    Raises
    ------
    om.AnalysisError
       If density is negative
    """
    def setup(self):
        self.add_input("ne", units="n20", desc="Electron density")
        self.add_input("Te", units="eV", desc="Electron temperature")
        self.add_output("logΛe",
                        lower=3,
                        ref=20,
                        desc="Coulomb collision logarithm for e⁻")
        self.c0 = 31.3

    def compute(self, inputs, outputs):
        ne20 = inputs["ne"]
        Te = inputs["Te"]

        if ne20 < 0:
            raise om.AnalysisError("Negative densities not allowed.")

        logLe = self.c0 - np.log(10**10 * ne20**(1 / 2) / Te)
        outputs["logΛe"] = logLe

    def setup_partials(self):
        self.declare_partials("logΛe", ["ne", "Te"])

    def compute_partials(self, inputs, J):
        J["logΛe", "ne"] = -1 / (2 * inputs["ne"])
        J["logΛe", "Te"] = 1 / inputs["Te"]


class CoulombLogarithmIons(om.ExplicitComponent):
    r"""Ion coulomb logarithm

    .. math::
        \log Λ_{ii} = 30 - \log(Z^3 \sqrt{n_i} / T_i^{3/2})

    Where ni is in per cubic meter and Ti is in eV.
    The formula is from :footcite:t:`sauter_neoclassical_1999`.

    Inputs
    ------
    ni : float
       n20, main ion density
    Ti : float
       eV, ion temperature
    Z : float
       Function of ion charge. Naively use Z_average.

    Outputs
    -------
    logΛi : float
       Ion coulomb logarithm

    Notes
    -----
    This computation may be somewhat specific for fast-particle slowing.
    Need to check.

    Sauter notes that these formulas are only for single-species plasmas, and
    that multi-species plasmas are out-of-scope.
    Here we use Z_average.
    """
    def setup(self):
        self.add_input("ni", units="n20", desc="Ion density")
        self.add_input("Ti", units="eV", desc="Ion temperature")
        self.add_input("Z", desc="Ion charge number")
        self.add_output("logΛi",
                        lower=0,
                        ref=20,
                        desc="Coulomb collision logarithm for ions")
        self.c0 = 30

    def compute(self, inputs, outputs):
        ni20 = inputs["ni"]
        Ti = inputs["Ti"]
        Z = inputs["Z"]
        logLi = self.c0 - np.log(Z**3 * 10**10 * ni20**(1 / 2) / Ti**(3 / 2))
        outputs["logΛi"] = logLi

    def setup_partials(self):
        self.declare_partials("logΛi", ["ni", "Ti", "Z"])

    def compute_partials(self, inputs, J):
        J["logΛi", "ni"] = -1 / (2 * inputs["ni"])
        J["logΛi", "Ti"] = 3 / (2 * inputs["Ti"])
        J["logΛi", "Z"] = -3 / inputs["Z"]
