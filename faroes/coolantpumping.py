from faroes.configurator import UserConfigurator

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials


class SimpleCoolantPumpingPower(om.ExplicitComponent):
    r"""

    Assumes it takes a constant fraction of the generated thermal power
    to pump the coolant. In general this should depend on the blanket
    details.

    Inputs
    ------
    P_thermal : float
        MW, Plasma primary thermal power

    Outputs
    -------
    P_pumps : float
        MW, pumping power
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        self.add_input("P_thermal", units="MW")
        self.add_output("P_pumps", units="MW", lower=0)

        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['machine'])
            self.c = ac(["coolant pumping system", "power factor"])
        else:
            self.c = 0.03

    def compute(self, inputs, outputs):
        outputs["P_pumps"] = self.c * inputs["P_thermal"]

    def setup_partials(self):
        c = self.c
        self.declare_partials("P_pumps", ["P_thermal"], val=c)


if __name__ == "__main__":
    uc = UserConfigurator()
    prob = om.Problem()

    prob.model = SimpleCoolantPumpingPower(config=uc)
    prob.setup(force_alloc_complex=True)

    prob.set_val('P_thermal', 100, units="MW")

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)
    prob.run_driver()
    all_inputs = prob.model.list_inputs(val=True)
    all_outputs = prob.model.list_outputs(val=True)
