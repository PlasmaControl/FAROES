import openmdao.api as om


class TFMagnetSet(om.ExplicitComponent):
    r"""Calculate quantities for a generic TF magnet set

    Inputs
    ------
    cross section : float
        m**2, Cross section of magnet
    arc length : float
        m, (Typical) perimeter of the magnet
    n_coil : float
        Number of coils.
    I_leg : float
        kA, Current per leg

    Outputs
    -------
    V_single : float
        m**3, Approximate material volume of one coil
    V_set : float
        m**3, Approximate material volume of the set
    conductor total: float
        kA * m, Total quantity of conductor
    """
    def setup(self):
        self.add_input("cross section",
                       units='m**2',
                       desc="Cross section of single magnet")
        self.add_input("arc length",
                       units='m',
                       desc="Typical perimeter of the magnet")
        self.add_input("I_leg", units='kA', desc="Current per leg")
        self.add_input("n_coil", desc="Number of coils in the set")

        self.add_output("V_single",
                        units='m**3',
                        ref=20,
                        desc="Material volume of one coil")
        self.add_output("V_set",
                        units='m**3',
                        ref=200,
                        desc="Material volume of the set")
        self.add_output("conductor total",
                        units='kA * m',
                        ref=1e6,
                        desc="Total quantity of conductor")

    def compute(self, inputs, outputs):
        cs = inputs["cross section"]
        l_arc = inputs["arc length"]
        n_coil = inputs["n_coil"]
        v_single = cs * l_arc
        v_set = v_single * n_coil

        outputs["V_single"] = v_single
        outputs["V_set"] = v_set
        outputs["conductor total"] = l_arc * n_coil * inputs["I_leg"]

    def setup_partials(self):
        self.declare_partials("V_single", ["cross section", "arc length"])
        self.declare_partials("V_set",
                              ["cross section", "arc length", "n_coil"])

        self.declare_partials("conductor total",
                              ["I_leg", "arc length", "n_coil"])

    def compute_partials(self, inputs, J):
        cs = inputs["cross section"]
        l_arc = inputs["arc length"]
        n_coil = inputs["n_coil"]
        i_leg = inputs["I_leg"]
        J["V_single", "cross section"] = l_arc
        J["V_single", "arc length"] = cs
        J["V_set", "cross section"] = l_arc * n_coil
        J["V_set", "arc length"] = cs * n_coil
        J["V_set", "n_coil"] = cs * l_arc
        J["conductor total", "n_coil"] = i_leg * l_arc
        J["conductor total", "arc length"] = i_leg * n_coil
        J["conductor total", "I_leg"] = l_arc * n_coil


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = TFMagnetSet()

    prob.setup()
    # prob.check_config(checks=['unconnected_inputs'])

    # initial values for design variables
    prob.set_val('I_leg', 5.1, 'MA')
    prob.set_val('arc length', 20, 'm')
    prob.set_val('cross section', 0.8, 'm**2')
    prob.set_val('n_coil', 18)

    prob.run_driver()

    prob.model.list_inputs(val=True, units=True, desc=True)
    prob.model.list_outputs(val=True, units=True, desc=True)
