from faroes.bremsstrahlung import Bremsstrahlung
from faroes.synchrotron import Synchrotron
from faroes.shapefactor import PeakingFactor
from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

import openmdao.api as om

SIMPLEBSZ = "brem-synch-simple-z"
SIMPLE = "simple"


class CoreRadiationProperties(om.Group):
    def initialize(self):
        self.options.declare('config', default=None)
        self.options.declare('bszoverride', default=False)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = self.options['config'].accessor(['plasma', 'radiation'])
        bszoverride = self.options['bszoverride']
        model = acc(['model'])

        if model == SIMPLEBSZ or bszoverride:
            rf = acc([SIMPLEBSZ, 'radiation factor'])
        elif model == SIMPLE:
            rf = acc([model, 'radiation factor'])
        else:
            raise ValueError(
                "Only 'simple' and 'simplebsz' models are implemented")
        ivc.add_output('radiation fraction', val=rf)
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
        Aspect ratio
    minor_radius : float
        m, Plasma horizontal minor radius 'a'
    κ : float
        Plasma elongation
    δ0 : float
        Triangularity of the LCFS
    Bt : float
        T, Toroidal field on axis
    P_heat : float
        MW, Total heating power into the core: α and external
    αn : float
        Exponent in density profile
    αT : float
        Exponent in temperature profile
    β : float
        Exponent in temperature profile
    n0 : float
        m**(-3), Electron density on axis
    T0 : float
        keV, Electron temperature on axis
    Z_eff : float
        Plasma effective charge
    Outputs
    -------
    P_rad : float
        MW, Core radiated power
    P_loss : float
        MW, Non-radiated power through the seperatrix.

    Notes
    -----
    f_rad = constant, loaded from config file

    .. math::

        P_\mathrm{rad} = P_\mathrm{Brems} + P_\mathrm{synch}
                       + P_\mathrm{core,imp,rad}
        P_\mathrm{core,imp,rad} = f_\mathrm{rad} * P_\mathrm{heat}
        P_\mathrm{loss} = P_\mathrm{heat} - P_\mathrm{rad}

    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           CoreRadiationProperties(config=config,
                                                   bszoverride=True),
                           promotes_outputs=[("radiation fraction", "f_rad")])

        self.add_subsystem("n_peaking_factor",
                           PeakingFactor(),
                           promotes_inputs=[
                               "A", ("a0", "minor_radius"), ("δ0", "δ"), "κ",
                               ("α", "αn")
                           ],
                           promotes_outputs=[("pf", "pf_n")])
        self.add_subsystem("n_axis",
                           om.ExecComp("n0= pf_n * n_mean",
                                       n0={"units": "m**(-3)"},
                                       pf_n={"units": None},
                                       n_mean={"units": "m**(-3)"}),
                           promotes_inputs=["pf_n", ("n_mean", "<n>")],
                           promotes_outputs=["n0"])

        self.add_subsystem("T_peaking_factor",
                           PeakingFactor(),
                           promotes_inputs=[
                               "A", ("a0", "minor_radius"), ("δ0", "δ"), "κ",
                               ("α", "αT")
                           ],
                           promotes_outputs=[("pf", "pf_T")])
        self.add_subsystem("T_axis",
                           om.ExecComp("T0= pf_T * T_mean",
                                       T0={"units": "keV"},
                                       pf_T={"units": None},
                                       T_mean={"units": "keV"}),
                           promotes_inputs=["pf_T", ("T_mean", "<T>")],
                           promotes_outputs=["T0"])

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
            "rad",
            om.ExecComp([
                "P_rad = P_brems + P_synch + P_coreimprad",
                "P_loss = P_heat - (P_brems + P_synch + P_coreimprad)"
            ],
                        P_rad={'units': 'MW'},
                        P_heat={'units': 'MW'},
                        P_loss={'units': 'MW'},
                        P_brems={'units': 'MW'},
                        P_synch={'units': 'MW'},
                        P_coreimprad={'units': 'MW'}),
            promotes_inputs=["P_heat", "P_brems", "P_synch", "P_coreimprad"],
            promotes_outputs=["P_rad", "P_loss"])

        # settings for parabolic profile sanity
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

    prob.set_val("A", 3.18)
    prob.set_val("minor_radius", 2.66, units='m')
    prob.set_val("κ", 1.8)
    prob.set_val("δ", 0.34)
    prob.set_val("Bt", 5.53, units="T")

    prob.set_val("Z_eff", 2.0)

    prob.set_val("P_heat", 1000)
    prob.set_val("<n>", 1.1, units="n20")
    prob.set_val("<T>", 15, units="keV")
    prob.set_val("αn", 0.5)
    prob.set_val("αT", 1.1)

    prob.run_driver()
    prob.model.list_inputs(values=True, print_arrays=True, units=True)
    prob.model.list_outputs(values=True, print_arrays=True)
