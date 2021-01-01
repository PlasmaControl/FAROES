from scipy.constants import mu_0
import openmdao.api as om


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
