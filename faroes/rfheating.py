import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor


class SimpleRFHeatingProperties(om.Group):
    r"""Helper for the SimpleRFHeating

    Loads from the configuration tree::

        h_cd:
          RF:
            power: <P>
            wall-plug efficiency: <eff>

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Outputs
    -------
    P : float
        MW, RF heating power absorbed in plasma
    eff : float
        Wall-plug efficiency
    """
    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["h_cd", "RF"])
        acc.set_output(ivc, f, "power", component_name="P", units="MW")
        acc.set_output(ivc, f, "wall-plug efficiency", component_name="eff")
        self.add_subsystem("ivc", ivc, promotes=["*"])


class SimpleRFHeating(om.Group):
    r"""RF heating that does not drive current

    Calculates the required auxilliary power

    .. math:: P_\mathrm{aux} = P / \mathrm{eff}

    where :math:`P` and 'eff' are loaded from the configuration tree
    by :class:`.SimpleRFHeatingProperties`.

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Outputs
    -------
    P : float
        MW, RF heating power absorbed in plasma
    eff : float
        Wall-plug efficiency
    P_aux : float
        MW, Wall-plug power
    """
    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("props",
                           SimpleRFHeatingProperties(config=config),
                           promotes=["*"])

        self.add_subsystem(
            "P_aux",
            om.ExecComp(
                "P_aux = P / eff",
                P_aux={
                    'units': 'MW',
                    'val': 0,
                    'desc': "Wall plug power for RF heating"
                },
                P={
                    'units': 'MW',
                    'val': 0,
                    'desc': "Plasma heating power"
                },
                eff={'desc': "Radiofrequency heating wall-plug efficiency"}),
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
