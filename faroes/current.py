from faroes.configurator import UserConfigurator, Accessor
import faroes.units  # noqa: F401
import openmdao.api as om
from scipy.constants import pi


class TokamakMagneticConfigurationLimitProperties(om.Group):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["plasma"])
        acc.set_output(ivc, f, "q_min")
        acc.set_output(ivc, f, "Greenwald fraction")
        self.add_subsystem("ivc", ivc, promotes=["*"])


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
        self.add_input("R0", units="m", val=1)
        self.add_input("L_pol", units="m")
        self.add_input("a", units="m")
        self.add_input("Ip", units="MA", val=1)
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


class LineAveragedDensity(om.ExplicitComponent):
    r"""Line-averaged plasma density.

    .. math::
       \bar{n} = f_{GW} I_p / (\pi a^2)

    Notes
    -----
    The line-averaged density is different from volume-averaged density.
    For a given density profile shape the difference is a constant factor.

    Inputs
    ------
    Ip : float
        MA, Plasma current
    a : float
        m, minor radius
    Greenwald fraction : float
        Fraction of the greenwald limit

    Output
    ------
    n_GW : float
        n20, Greenwald density
    n_bar : float
        n20, line-averaged electron density
    """

    def setup(self):
        self.add_input("Ip", units="MA")
        self.add_input("a", units="m")
        self.add_input("Greenwald fraction")
        tiny = 1e-6
        self.add_output("n_GW", units="n20", lower=tiny, ref=1)
        self.add_output("n_bar", units="n20", lower=tiny, ref=1)

    def compute(self, inputs, outputs):
        Ip = inputs["Ip"]
        a = inputs["a"]
        n_GW = Ip / (pi * a**2)
        f_GW = inputs["Greenwald fraction"]
        outputs["n_GW"] = n_GW
        outputs["n_bar"] = n_GW * f_GW

    def setup_partials(self):
        self.declare_partials(["n_GW", "n_bar"], ["Ip", "a"])
        self.declare_partials(["n_bar"], ["Greenwald fraction"])

    def compute_partials(self, inputs, J):
        Ip = inputs["Ip"]
        a = inputs["a"]
        f_GW = inputs["Greenwald fraction"]
        J["n_GW", "Ip"] = 1 / (pi * a**2)
        J["n_GW", "a"] = -2 * Ip / (pi * a**3)
        J["n_bar", "Ip"] = f_GW * J["n_GW", "Ip"]
        J["n_bar", "a"] = f_GW * J["n_GW", "a"]
        J["n_bar", "Greenwald fraction"] = Ip / (pi * a**2)


class TotalPlasmaCurrent(om.ExplicitComponent):
    r"""

    Inputs
    ------
    I_BS : float
        MA, bootstrap current
    I_NBI : float
        MA, neutral-beam-driven current
    I_RF : float
        MA, RF-driven current
    I_ohmic : float
        MA, Ohmic current

    Outputs
    -------
    Ip : float
        MA, plasma current
    f_BS : float
        Bootstrap fraction

    Notes
    -----
    The bootstrap fraction computation is a 're-computation' for Menard's
    model, where f_BS is calculated first and the I_BS from that.
    This could be a nice diagnostic in case the bootstrap current computation
    is changed to be more physically-based in the future.
    """

    def setup(self):
        self.add_input("I_BS", units="MA", val=5)
        self.add_input("I_NBI", units="MA", val=5)
        self.add_input("I_RF", units="MA", val=0)
        self.add_input("I_ohmic", units="MA", val=0)
        tiny = 1e-3
        self.add_output("Ip", units="MA", lower=tiny, val=10, ref=10)
        self.add_output("f_BS", units="MA", lower=0, val=0.5, upper=1, ref=1)

    def compute(self, inputs, outputs):
        Ip = inputs["I_BS"] + inputs["I_NBI"] + \
            inputs["I_RF"] + inputs["I_ohmic"]
        outputs["Ip"] = Ip
        outputs["f_BS"] = inputs["I_BS"] / Ip

    def setup_partials(self):
        self.declare_partials("Ip", ["I_BS", "I_NBI", "I_RF", "I_ohmic"],
                              val=1)
        self.declare_partials("f_BS", ["I_BS", "I_NBI", "I_RF", "I_ohmic"])

    def compute_partials(self, inputs, J):
        I_BS = inputs["I_BS"]
        I_NBI = inputs["I_NBI"]
        I_RF = inputs["I_RF"]
        I_ohmic = inputs["I_ohmic"]
        Ip = I_BS + I_NBI + I_RF + I_ohmic
        J["f_BS", "I_BS"] = (I_NBI + I_RF + I_ohmic) / (Ip**2)
        J["f_BS", "I_NBI"] = -I_BS / Ip**2
        J["f_BS", "I_RF"] = -I_BS / Ip**2
        J["f_BS", "I_ohmic"] = -I_BS / Ip**2


class CurrentAndSafetyFactor(om.Group):
    r"""

    Inputs
    ------
    I_BS : float
        MA, bootstrap current
    I_NBI : float
        MA, neutral-beam-driven current
    I_RF : float
        MA, RF-driven current
    I_ohmic : float
        MA, Ohmic current
    R0 : float
        m, Major radius
    a : float
        m, minor radius
    L_pol : float
        m, toroidal circumference
    Bt : float
        T, Toroidal field

    Outputs
    -------
    Ip : float
        MA, total plasma current
    I/aB: float
        Factor used in some scaling laws. Incidental output.
    q_star : float
        Cylindrical safety factor
    Greenwald fraction : float
        Specified Greenwald fraction
    n_GW : float
        n20, Greenwald density
    n_bar : float
        n20, line-averaged electron density
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem(
            "props",
            TokamakMagneticConfigurationLimitProperties(config=config),
            promotes_outputs=["*"])
        self.add_subsystem("current", TotalPlasmaCurrent(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])
        self.add_subsystem("qcyl",
                           QCylindrical(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])
        self.add_subsystem("ngw",
                           LineAveragedDensity(),
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()
    prob.model = CurrentAndSafetyFactor(config=uc)
    prob.setup()

    prob.set_val("Bt", 2.094, units="T")
    prob.set_val("I_NBI", 7.67, units="MA")
    prob.set_val("I_BS", 7.00, units="MA")
    prob.set_val("a", 1.875, units="m")
    prob.set_val("L_pol", 23.3, units="m")
    prob.set_val("R0", 3.0, units="m")

    prob.run_driver()
    prob.model.list_inputs(values=True, print_arrays=True)
    prob.model.list_outputs(values=True, print_arrays=True)
