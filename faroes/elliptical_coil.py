import openmdao.api as om
import faroes.util as util
from scipy.constants import pi
from faroes.configurator import Accessor


class TFSetProperties(om.Group):
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["magnet_geometry", "profile"])
        acc.set_output(ivc, f, "elongation_multiplier")
        self.add_subsystem("ivc", ivc, promotes=["*"])


class SimpleEllipticalTFSet(om.ExplicitComponent):
    r"""Simple elliptical TF coil set

    Inputs
    ------
    R0 : float
        m, plasma major radius
    r1 : float
        m, inboard leg radius
    r2 : float
        m, outboard leg radius
    cross section : float
        m**2, inboard horizontal cross section
    κ : float
        plasma's kappa
    elongation_multiplier : float
        the TF coils may be less elongated than the plasma
    n_coil : int
        number of TF coils

    Outputs
    -------
    half-width : float
        m, horizontal semi-axis of the ellipse
    half-height : float
        m, vertical semi-axis of the ellipse
    arc length: float
        m, average perimeter of the magnet
    V_single : float
        m**3, material volume of one coil
    V_set : float
        m**3, material volume of the set
    V_enc : float
        m**3, magnetized volume enclosed by the set
    """

    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        self.add_input("R0", units="m", desc="Plasma major radius")
        self.add_input("r1",
                       units="m",
                       desc="Inboard TF leg average conductor radius")
        self.add_input("r2",
                       units="m",
                       desc="Outboard TF leg average conductor radius")
        self.add_input("cross section",
                       units="m**2",
                       desc="Inboard horizontal cross section")
        self.add_input("κ", desc="Plasma's elongation")
        self.add_input(
            "elongation_multiplier",
            desc="The TF coils may be less elongated than the plasma")
        self.add_input('n_coil', 18, desc='number of coils')

        self.add_output("half-width",
                        units="m", lower=0,
                        desc="Average semi-minor axis of the magnet")
        self.add_output("half-height",
                        units="m", lower=0,
                        desc="Average semi-major axis of the magnet")
        self.add_output("arc length",
                        units="m", lower=0,
                        desc="Average perimeter of the magnet")
        self.add_output("V_single",
                        units="m**3", lower=0,
                        desc="Material volume of one coil")
        self.add_output("V_set",
                        units="m**3", lower=0,
                        desc="Material volume of the set")
        V_enc_ref = 1e3
        self.add_output("V_enc",
                        units="m**3",
                        lower=0,
                        ref=V_enc_ref,
                        desc="Magnetized volume enclosed by the set")

    def compute(self, inputs, outputs):
        R0 = inputs["R0"]
        r1 = inputs["r1"]
        r2 = inputs["r2"]
        half_w = (r2 - r1) / 2
        outputs["half-width"] = half_w

        κ = inputs["κ"]
        elong_mult = inputs["elongation_multiplier"]
        elong = κ * elong_mult

        half_height = half_w * elong
        outputs["half-height"] = half_height

        arc_length = util.ellipse_perimeter_ramanujan(half_w, half_height)
        outputs["arc length"] = arc_length

        v_single = inputs["cross section"] * arc_length
        outputs["V_single"] = v_single

        outputs["V_set"] = inputs["n_coil"] * v_single

        v_enc = (pi * half_w * half_height) * (2 * pi * R0)
        outputs["V_enc"] = v_enc

    def setup_partials(self):
        self.declare_partials('half-width', ['r1', 'r2'])
        self.declare_partials('half-height',
                              ['r1', 'r2', 'κ', 'elongation_multiplier'])
        self.declare_partials('arc length',
                              ['r1', 'r2', 'κ', 'elongation_multiplier'])
        self.declare_partials(
            'V_single',
            ['r1', 'r2', 'κ', 'elongation_multiplier', 'cross section'])
        self.declare_partials('V_set',
                              ['r1', 'r2', 'κ', 'n_coil',
                               'elongation_multiplier', 'cross section'])
        self.declare_partials('V_enc',
                              ['r1', 'r2', 'κ', 'elongation_multiplier', "R0"])

    def compute_partials(self, inputs, J):
        n_coil = inputs['n_coil']
        R0 = inputs["R0"]
        r1 = inputs["r1"]
        r2 = inputs["r2"]
        κ = inputs["κ"]

        half_width = (r2 - r1) / 2
        J["half-width", "r1"] = -1 / 2
        J["half-width", "r2"] = 1 / 2

        elong_mult = inputs["elongation_multiplier"]
        elong = κ * elong_mult

        half_height = half_width * elong

        J["half-height", "r1"] = -elong / 2
        J["half-height", "r2"] = +elong / 2
        J["half-height", "κ"] = half_width * elong_mult
        J["half-height", "elongation_multiplier"] = half_width * κ

        # arc length, also known as perimeter
        Pe = util.ellipse_perimeter_ramanujan(half_width, half_height)
        dPe = util.ellipse_perimeter_ramanujan_derivatives(
            half_width, half_height)

        J["arc length",
          "r1"] = dPe['a'] * J["half-width",
                               "r1"] + dPe['b'] * J["half-height", "r1"]
        J["arc length",
          "r2"] = dPe['a'] * J["half-width",
                               "r2"] + dPe['b'] * J["half-height", "r2"]
        J["arc length", "κ"] = dPe['b'] * J["half-height", "κ"]
        J["arc length",
          "elongation_multiplier"] = dPe['b'] * J["half-height",
                                                  "elongation_multiplier"]

        A_cs = inputs["cross section"]
        J["V_single", "r1"] = A_cs * J["arc length", "r1"]
        J["V_single", "r2"] = A_cs * J["arc length", "r2"]
        J["V_single", "κ"] = A_cs * J["arc length", "κ"]
        J["V_single",
          "elongation_multiplier"] = A_cs * J["arc length",
                                              "elongation_multiplier"]
        J["V_single", "cross section"] = Pe

        arc_length = util.ellipse_perimeter_ramanujan(half_width, half_height)
        v_single = inputs["cross section"] * arc_length
        J["V_set", "n_coil"] = v_single
        J["V_set", "r1"] = n_coil * J["V_single", "r1"]
        J["V_set", "r2"] = n_coil * J["V_single", "r2"]
        J["V_set", "κ"] = n_coil * J["V_single", "κ"]
        J["V_set",
          "elongation_multiplier"] = n_coil * J["V_single",
                                                "elongation_multiplier"]
        J["V_set", "cross section"] = n_coil * J["V_single", "cross section"]

        J["V_enc", "r1"] = -elong * pi**2 * R0 * (r2 - r1)
        J["V_enc", "r2"] = +elong * pi**2 * R0 * (r2 - r1)
        J["V_enc", "κ"] = elong_mult * pi**2 * R0 * (r2 - r1)**2 / 2
        J["V_enc", "elongation_multiplier"] = κ * pi**2 * R0 * (r2 - r1)**2 / 2
        J["V_enc", "R0"] = elong * pi**2 * (r2 - r1)**2 / 2


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = SimpleEllipticalTFSet()

    prob.setup()

    prob.set_val("elongation_multiplier", 0.7)
    prob.set_val("κ", 2.73977961)
    prob.set_val("R0", 3)
    prob.set_val("r2", 8.168)
    prob.set_val("r1", 0.26127)
    prob.set_val("cross section", 0.052)
    prob.set_val("n_coil", 18)

    prob.run_driver()
    all_outputs = prob.model.list_inputs(val=True, desc=True)
    all_outputs = prob.model.list_outputs(val=True, desc=True)
