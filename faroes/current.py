from faroes.configurator import UserConfigurator, Accessor
import openmdao.api as om
from scipy.constants import pi


class MagneticConfigurationProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        acc = Accessor(self.options['config'])
        f = acc.accessor(["plasma"])
        acc.set_output(self, f, "q_min")


class QCylindrical(om.ExplicitComponent):
    r"""q_star, also known as q* or q_cyl or cylindrical safety factor

    .. math::
        q^* = \epsilon (1 + \kappa^2) \pi a B / \mu_0 I

    The 1 + kappa squared is an approximation for part of the poloidal
    circumference, so I use that instead.

    .. math::
       q^* = 5 L_\mathrm{pol}^2 B / (4 \pi^2 R (I/10^6))

    Inputs
    ------
    R0 : float
        m, Major radius
    a : float
        m, minor radius
    L_pol : float
        m, toroidal circumference
    Ip : float
        MA, Plasma current
    Bt : float
        T, Toroidal field

    Outputs
    -------
    I/aB: float
        Factor used in some scaling laws. Incidental output.
    q_star : float
        Cylindrical safety factor

    References
    ----------
    [1] Menard, J. E.; Bell, M. G.; Bell, R. E.; Gates, D. A.; Kaye, S. M.;
    LeBlanc, B. P.; Maingi, R.; Sabbagh, S. A.; Soukhanovskii, V.; Stutman, D.;
    NSTX National Research Team.
    Aspect Ratio Scaling of Ideal No-Wall Stability Limits in High Bootstrap
    Fraction Tokamak Plasmas.
    Physics of Plasmas 2004, 11 (2), 639–646.
    https://doi.org/10.1063/1.1640623.
    """
    def setup(self):
        self.add_input("R0", units="m")
        self.add_input("L_pol", units="m")
        self.add_input("a", units="m")
        self.add_input("Ip", units="MA")
        self.add_input("Bt", units="T")
        self.add_output("I/aB", lower=0)
        self.add_output("q_star", lower=0)

    def compute(self, inputs, outputs):
        R = inputs["R0"]
        Lpol = inputs["L_pol"]
        Bt = inputs["Bt"]
        Ip = inputs["Ip"]
        a = inputs["a"]
        outputs["q_star"] = 5 * Lpol**2 * Bt / (4 * pi**2 * R * Ip)
        outputs["I/aB"] = Ip / (a * Bt)

    def setup_partials(self):
        self.declare_partials(["I/aB", "q_star"], ["Ip", "Bt"])
        self.declare_partials(["I/aB"], ["a"])
        self.declare_partials(["q_star"], ["R0", "L_pol"])

    def compute_partials(self, inputs, J):
        R = inputs["R0"]
        Lpol = inputs["L_pol"]
        Bt = inputs["Bt"]
        Ip = inputs["Ip"]
        a = inputs["a"]
        J["q_star", "L_pol"] = 10 * Lpol * Bt / (4 * pi**2 * R * Ip)
        J["q_star", "Bt"] = 5 * Lpol**2 / (4 * pi**2 * R * Ip)
        J["q_star", "R0"] = -5 * Lpol**2 * Bt / (4 * pi**2 * R**2 * Ip)
        J["q_star", "Ip"] = -5 * Lpol**2 * Bt / (4 * pi**2 * R * Ip**2)
        J["I/aB", "Ip"] = 1 / (a * Bt)
        J["I/aB", "a"] = -Ip / (a**2 * Bt)
        J["I/aB", "Bt"] = -Ip / (a * Bt**2)


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()
    prob.model = QCylindrical()
    prob.setup()

    prob.set_val("Bt", 2.094, units="T")
    prob.set_val("Ip", 14.67, units="MA")
    prob.set_val("a", 1.875, units="m")
    prob.set_val("L_pol", 23.3, units="m")
    prob.set_val("R0", 3.0, units="m")

    prob.run_driver()
    prob.model.list_inputs(values=True, print_arrays=True)
    prob.model.list_outputs(values=True, print_arrays=True)
