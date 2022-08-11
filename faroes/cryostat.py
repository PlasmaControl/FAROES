import openmdao.api as om
from scipy.constants import pi


class SimpleCryostat(om.ExplicitComponent):
    r"""Calculates cryostat height and volume

    The cryostat height is a constant multiple, typically :math:`m=2`,
    of the toroidal field coil total height:

    .. math:: h_\mathrm{cryostat} = m\,\left(2\cdot\frac{1}{2}h_{TH}\right).

    The cryostat volume is that of a cylinder,

    .. math::
       V_\mathrm{cryostat} = π\,R_\mathrm{out}^2 h_\mathrm{cryostat}.

    Options
    -------
    config : UserConfigurator
        Configuration tree.

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

    Notes
    -----
    The multiplier to find the cryostat height is loaded from the configuration
    tree::

      machine:
        cryostat:
          TF height multiple: <value>

    If the configuration tree is not loaded, tf_height_multiple is 2.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        if self.options['config'] is not None:
            config = self.options['config'].accessor(["machine", "cryostat"])
            self.tf_height_multiple = config(["TF height multiple"])
        else:
            self.tf_height_multiple = 2

        self.add_input("R_out", units="m", desc="Cryostat outer radius")
        self.add_input("TF half-height",
                       units="m",
                       desc="Half the outer full height of the TF coils")

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
    all_outputs = prob.model.list_inputs(val=True, desc=True)
    all_outputs = prob.model.list_outputs(val=True, desc=True)
