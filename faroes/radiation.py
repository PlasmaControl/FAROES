from faroes.configurator import UserConfigurator, Accessor
import faroes.units  # noqa: F401

import openmdao.api as om


class CoreRadiationProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        acc = Accessor(self.options['config'])
        f = acc.accessor(["plasma"])
        acc.set_output(self, f, "radiation fraction")


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


class SimpleRadiation(om.Group):
    r"""Simplest division of power into P_rad and P_loss

    Notes
    -----
    f_rad = constant, loaded from config file

    .. math::

        P_\mathrm{rad} = P_\mathrm{heat}* (1 - f_\mathrm{rad})
        P_\mathrm{loss} = P_\mathrm{heat}* (1 - f_\mathrm{rad})

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


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()
    prob.model = SimpleRadiation(config=uc)
    prob.setup()

    prob.set_val("P_heat", 100)

    prob.run_driver()
    prob.model.list_inputs(values=True, print_arrays=True)
    prob.model.list_outputs(values=True, print_arrays=True)
