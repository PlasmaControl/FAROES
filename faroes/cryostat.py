from faroes.configurator import Accessor
import openmdao.api as om
from scipy.constants import pi

class SimpleCryostat(om.ExplicitComponent):
    r"""Simple cryostat model

    Inputs
    ------
    R_out : float
        m, Outer radius of cryostat.
    TF half-height : float
        m, Half-height of the TF coils

    Outputs
    -------
    h : float
        m, height
    V : float
        m**3, volume
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            config = self.options['config'].accessor(["machine", "cryostat"])
            self.tf_height_multiple = config(["TF height multiple"])
        else:
            self.tf_height_multiple = 2

        self.add_input("R_out", units="m", desc="Outer radius")
        self.add_input("TF half-height", units="m", desc="Outer radius")

        self.add_output("V", units="m**3", desc="Cryostat total volume")
        self.add_output("h", units="m", desc="Cryostat height")

    def compute(self, inputs, outputs):
        R = inputs["R_out"]
        tf_hh = inputs["TF half-height"]

        h = tf_hh * 2 * self.tf_height_multiple
        outputs["h"] = h
        outputs["V"] = pi * R**2 * h

    def setup_partials(self):
        self.declare_partials('h', ['TF half-height'])
        self.declare_partials('V', ['R_out', 'TF half-height'])

    def compute_partials(self, inputs, J):
        R = inputs["R_out"]
        tf_hh = inputs["TF half-height"]

        h = tf_hh * 2 * self.tf_height_multiple
        J["h", "TF half-height"] = 2 * self.tf_height_multiple
        J["V", "TF half-height"] = pi * R**2 * J["h", "TF half-height"]
        J["V", "R_out"] = 2 * pi * R * h

if __name__ == "__main__":
    prob = om.Problem()

    prob.model = SimpleCryostat()

    prob.setup()

    prob.set_val("R_out", 1)
    prob.set_val("TF half-height", 1)

    prob.run_driver()
    all_outputs = prob.model.list_outputs(values=True)
