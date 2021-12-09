import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor


class SimpleRFHeatingProperties(om.Group):
    r"""Helper for the SimpleNBISource

    Outputs
    -------
    P : float
        MW, Neutral beam power to plasma
    eff : float
        Wall-plug efficiency
    P_aux : float
        MW, Wall-plug power
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["h_cd", "RF"])
        acc.set_output(ivc, f, "power", component_name="P", units="MW")
        acc.set_output(ivc, f, "wall-plug efficiency", component_name="eff")
        self.add_subsystem("ivc", ivc, promotes=["*"])


class SimpleRFHeating(om.Group):
    r"""RF heating that does not drive current, but might be more efficient than
    NBI heating.

    Outputs
    -------
    P : float
        MW, Neutral beam power to plasma
    eff : float
        Wall-plug efficiency
    P_aux : float
        MW, Wall-plug power
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("props",
                           SimpleRFHeatingProperties(config=config),
                           promotes=["*"])

        self.add_subsystem("P_aux",
                           om.ExecComp("P_aux = P / eff",
                                       P_aux={
                                           'units': 'MW',
                                           'val': 0
                                       },
                                       P={
                                           'units': 'MW',
                                           'val': 0
                                       }),
                           promotes_inputs=["P", "eff"],
                           promotes_outputs=["P_aux"])


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model = SimpleRFHeating(config=uc)

    prob.setup()
    prob.set_val("P", 10, units="MW")
    prob.run_driver()

    prob.model.list_inputs(val=True)
    prob.model.list_outputs(val=True)
