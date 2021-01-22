import openmdao.api as om
import faroes.util as util


class BetaNTotal(om.ExplicitComponent):
    r"""Total β_N from a scaling law and a multiplier

    Inputs
    ------
    A : float
        Aspect ratio

    Outputs
    -------
    β_N : float
        normalized beta from a scaling law

    β_N total : float
        the total normalized beta

    Reference
    ---------
    Physics of Plasmas 11, 639 (2004);
    https://doi.org/10.1063/1.1640623

    No-wall limit, with 50% bootstrap fraction
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['fits'])
            self.β_ε_scaling_constants = ac(
                ["no-wall β_N scaling with ε", "constants"])
            self.β_N_multiplier = ac(["β_N multiplier"])

        self.add_input("A", desc="Aspect Ratio")
        self.add_output("β_N", units="m * T / MA", desc="Normalized beta")
        self.add_output("β_N total",
                        units="m * T / MA",
                        desc="Normalized beta, total")

    def β_N_scaling(self, A):
        """Estimated β_N from A

        Parameters
        ----------
        A: float
            aspect ratio of the plasma

        Returns
        -------
        β_N : float
            Normalized total pressure

        """
        const = self.β_ε_scaling_constants
        b = const[0]
        c = const[1]
        d = const[2]
        return b + c / (A**d)

    def compute(self, inputs, outputs):
        A = inputs["A"]
        β_N = self.β_N_scaling(A)
        outputs["β_N"] = β_N
        outputs["β_N total"] = β_N * self.β_N_multiplier

    def setup_partials(self):
        self.declare_partials(["β_N", "β_N total"], "A")

    def compute_partials(self, inputs, J):
        const = self.β_ε_scaling_constants
        A = inputs["A"]
        b = const[0]
        c = const[1]
        d = const[2]
        J["β_N", "A"] = -A**(-d - 1) * c * d
        scale = self.β_N_multiplier
        J["β_N total", "A"] = scale * J["β_N", "A"]


class BetaT(om.ExplicitComponent):
    def setup(self):
        self.add_input("Ip", units="MA")
        self.add_input("Bt", units="T")
        self.add_input("a", units="m")
        self.add_input("β_N total", units="m * T / MA")
        βt_ref = 0.02
        self.add_output("βt",
                        lower=0,
                        upper=1,
                        ref=βt_ref,
                        desc="Toroidal beta")

    def compute(self, inputs, outputs):
        Ip = inputs["Ip"]
        Bt = inputs["Bt"]
        a = inputs["a"]
        βN_tot = inputs["β_N total"]
        βt = Ip * βN_tot / (Bt * a)
        outputs["βt"] = βt

    def setup_partials(self):
        self.declare_partials(["βt"], ["Ip", "Bt", "a", "β_N total"])

    def compute_partials(self, inputs, J):
        Ip = inputs["Ip"]
        Bt = inputs["Bt"]
        a = inputs["a"]
        βN_tot = inputs["β_N total"]
        J["βt", "Ip"] = βN_tot / (Bt * a)
        J["βt", "β_N total"] = Ip / (Bt * a)
        J["βt", "Bt"] = -Ip * βN_tot / (Bt**2 * a)
        J["βt", "a"] = -Ip * βN_tot / (Bt * a**2)


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = BetaNTotal()

    prob.model.β_N_multiplier = 1.1
    prob.model.β_ε_scaling_constants = [3.12, 3.5, 1.7]
    prob.setup()

    prob.set_val('A', 1.6)

    prob.run_driver()
    prob.model.list_outputs()
