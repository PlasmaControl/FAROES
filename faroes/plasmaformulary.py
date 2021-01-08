import openmdao.api as om

from scipy.constants import mu_0
import numpy as np


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


class AverageIonMass(om.ExplicitComponent):
    r"""Average ion mass

    .. math::

       \overline{A_i} = \sum_i n_i A_i / \sum n_i

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
        self.add_output("A_bar", units="u")

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
