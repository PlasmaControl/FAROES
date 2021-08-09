from faroes.bremsstrahlung import Bremsstrahlung
from faroes.synchrotron import Synchrotron
from faroes.configurator import UserConfigurator, Accessor
import faroes.units  # noqa: F401

import openmdao.api as om


class CoreRadiationProperties(om.Group):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["plasma"])
        acc.set_output(ivc, f, "radiation fraction")
        self.add_subsystem("ivc", ivc, promotes=["*"])


class TrivialRadiation(om.ExplicitComponent):
    r"""Constant fraction of core power

    Inputs
    ------
    P_heat : float
        MW, total heating power into the core: alpha and external
    f_rad : float
        Radiation fraction

    Outputs
    -------
    P_rad : float
        MW, Radiated power
    P_loss : float
        MW, power lost down the thermal gradient.
            Non-radiated power.
    """
    def setup(self):
        self.add_input("P_heat", units="MW")
        self.add_input("f_rad")
        P_ref = 100
        self.add_output("P_rad", ref=P_ref, lower=0.1, units="MW")
        self.add_output("P_loss", ref=P_ref, lower=0.1, units="MW")

    def compute(self, inputs, outputs):
        P_heat = inputs["P_heat"]
        f_rad = inputs["f_rad"]

        outputs["P_rad"] = f_rad * P_heat
        outputs["P_loss"] = (1 - f_rad) * P_heat

    def setup_partials(self):
        self.declare_partials(["P_rad", "P_loss"], ["P_heat", "f_rad"])

    def compute_partials(self, inputs, J):
        P_heat = inputs["P_heat"]
        f_rad = inputs["f_rad"]
        J["P_rad", "P_heat"] = f_rad
        J["P_rad", "f_rad"] = P_heat
        J["P_loss", "P_heat"] = (1 - f_rad)
        J["P_loss", "f_rad"] = -P_heat


class TrivialCoreImpurityRadiation(om.ExplicitComponent):
    r"""Core impurity radiation as a constant fraction of heating power

    Inputs
    ------
    P_heat : float
        MW, total heating power into the core: alpha and external
    f_rad : float
        Radiation fraction

    Outputs
    -------
    P_coreimprad : float
        MW, Radiated power
    """
    def setup(self):
        self.add_input("P_heat", units="MW")
        self.add_input("f_rad")
        P_ref = 100
        self.add_output("P_coreimprad", ref=P_ref, lower=0.1, units="MW")

    def compute(self, inputs, outputs):
        P_heat = inputs["P_heat"]
        f_rad = inputs["f_rad"]

        outputs["P_coreimprad"] = f_rad * P_heat

    def setup_partials(self):
        self.declare_partials(["P_coreimprad"], ["P_heat", "f_rad"])

    def compute_partials(self, inputs, J):
        P_heat = inputs["P_heat"]
        f_rad = inputs["f_rad"]
        J["P_coreimprad", "P_heat"] = f_rad
        J["P_coreimprad", "f_rad"] = P_heat


class SimpleRadiation(om.Group):
    r"""Simplest division of power into P_rad and P_loss

    Notes
    -----
    f_rad = constant, loaded from config file

    .. math::

        P_\mathrm{rad} = f_\mathrm{rad}* P_\mathrm{heat}
        P_\mathrm{loss} = (1 - f_\mathrm{rad}) * P_\mathrm{heat}

    Inputs
    ------
    P_heat : float
        MW, total heating power into the core: alpha and external

    Outputs
    -------
    P_rad : float
        MW, Radiated power
    P_loss : float
        MW, power lost down the thermal gradient.
            Non-radiated power.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           CoreRadiationProperties(config=config),
                           promotes_outputs=[("radiation fraction", "f_rad")])
        self.add_subsystem("rad", TrivialRadiation(), promotes=["*"])


class SimpleBSZRadiation(om.Group):
    r"""Analytic profiles of Brems, Synch; trivial core impurity

    Inputs
    ------
    A : float
        None, Aspect ratio (R0 / a0)
    δ0 : float
        None, Triangularity of border curve of plasma distribution
    κ : float
        None, Elongation of plasma distribution shape
    αn : float
        None, Exponent in density profile
    αT : float
        None, Exponent in temperature profile
    β : float
        None, Exponent in temperature profile
    ρpedn : float
        None, value of normalized radius at density pedestal top
    ρpedT : float
        None, value of normalized radius at temperature pedestal top
    n0 : float
        m**(-3), Electron density on axis
    nped : float
        m**(-3), density at pedestal top
    n1 : float
        m**(-3), density at separatix
    T0 : float
        keV, temperature at center
    Tped : float
        keV, temperature at pedestal top
    T1 : float
        keV, temperature at separatix

    Notes
    -----
    f_rad = constant, loaded from config file

    .. math::

        P_\mathrm{rad} = P_\mathrm{Brems} + P_\mathrm{synch}
                       + P_\mathrm{core,imp,rad}
        P_\mathrm{core,imp,rad} = f_\mathrm{rad} * P_\mathrm{heat}
        P_\mathrm{loss} = P_\mathrm{heat} - P_\mathrm{rad}

    Inputs
    ------
    P_heat : float
        MW, total heating power into the core: alpha and external

    Outputs
    -------
    P_rad : float
        MW, Radiated power
    P_loss : float
        MW, power lost down the thermal gradient.
            Non-radiated power.
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           CoreRadiationProperties(config=config),
                           promotes_outputs=[("radiation fraction", "f_rad")])
        self.add_subsystem("brems",
                           Bremsstrahlung(profile='parabolic',
                                          triangularity='constant'),
                           promotes_inputs=[
                               "A", ("a0", "minor_radius"), ("δ0", "δ"), "κ",
                               ("Zeff", "Z_eff"), "n0", "T0", ("β", "βT"),
                               "n1", "T1", "nped", "Tped", "αn", "αT"
                           ],
                           promotes_outputs=[("P", "P_brems")])

        self.add_subsystem("synch",
                           Synchrotron(),
                           promotes_inputs=[
                               "A", ("a0", "minor_radius"), "δ", "κ", "αn",
                               "αT", "Bt", "n0", "r", "T0", "βT"
                           ],
                           promotes_outputs=[("P", "P_synch")])
        self.add_subsystem("coreimprad",
                           TrivialCoreImpurityRadiation(),
                           promotes_inputs=["f_rad", "P_heat"],
                           promotes_outputs=["P_coreimprad"])

        self.add_subsystem(
            "P_rad",
            om.ExecComp(["P_rad = P_brems + P_synch + P_coreimprad"],
                        P_rad={'units': 'MW'},
                        P_brems={'units': 'MW'},
                        P_synch={'units': 'MW'},
                        P_coreimprad={'units': 'MW'}),
            promotes=["*"],
        )
        self.add_subsystem(
            "P_loss",
            om.ExecComp(["P_loss = P_heat - P_rad"],
                        P_rad={'units': 'MW'},
                        P_loss={'units': 'MW'},
                        P_heat={'units': 'MW'}),
            promotes=["*"],
        )

        self.set_input_defaults("A", 2.0)
        self.set_input_defaults("δ", 0.0)
        self.set_input_defaults("n0", 1.1, units='n20')
        self.set_input_defaults("αn", 0.5)
        self.set_input_defaults("αT", 1.0)
        self.set_input_defaults("r", 0.2)
        self.set_input_defaults("βT", 2)  # profile shape; NOT plasma beta
        self.set_input_defaults("n1", 0)
        self.set_input_defaults("T1", 0)
        self.set_input_defaults("nped", 0)
        self.set_input_defaults("Tped", 0)


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()
    prob.model = SimpleBSZRadiation(config=uc)
    prob.setup()

    prob.set_val("P_heat", 1000)
    prob.set_val("A", 3.18)
    prob.set_val("minor_radius", 2.66, units='m')
    prob.set_val("κ", 1.8)
    prob.set_val("Z_eff", 2.0)
    prob.set_val("δ", 0.34)

    prob.set_val("n0", 1.1, units="n20")
    prob.set_val("T0", 30, units="keV")
    prob.set_val("αn", 0.5)
    prob.set_val("αT", 1.1)
    prob.set_val("Bt", 5.53, units="T")

    prob.run_driver()
    prob.model.list_inputs(values=True, print_arrays=True, units=True)
    # prob.model.list_outputs(values=True, print_arrays=True)
