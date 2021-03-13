import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor

from faroes.blanket import MenardInboardBlanketFit
from faroes.blanket import MenardInboardShieldFit


class RadialBuildProperties(om.Group):
    r"""Helper for the Menard ST radial build
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["radial_build", "inboard"])
        acc.set_output(ivc,
                       f,
                       "vv shielding thickness",
                       component_name="Ib WC VV shield thickness",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "fw thickness",
                       component_name="Ib FW thickness",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "SOL width",
                       component_name="Ib SOL width",
                       units="m")

        f = acc.accessor(["radial_build", "outboard"])
        acc.set_output(ivc,
                       f,
                       "SOL width",
                       component_name="Ob SOL width",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "blanket thickness",
                       component_name="Ob blanket thickness",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "access thickness",
                       component_name="Ob access thickness",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "vv thickness",
                       component_name="Ob vv thickness",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "shield thickness",
                       component_name="Ob shield thickness",
                       units="m")
        acc.set_output(ivc, f, "TF-cryostat thickness", units="m")
        self.add_subsystem("ivc", ivc, promotes_outputs=["*"])


class MenardSTInboardRadialBuild(om.ExplicitComponent):
    r"""Inboard radial build, to plasma outboard

    Inputs
    ------
    CS R_max : float
        m, CS outer radius

    TF R_max : float
        m, Inboard TF leg maximum radius
    WC VV shield thickness : float
        m, Tungsten-carbide neutron shield b/w vacuum vessel shells
    WC shield thickness : float
        m, Tungsten-carbide neutron shield thickness
    blanket thickness : float
        m, Inboard blanket thickness
    FW thickness : float
        m, First wall thickness
    SOL width : float
        m, Scrape-off-layer width
    a : float
        m, Plasma minor radius

    Outputs
    -------
    TF R_min : float
        m, Inboard TF leg inner radius

    VV R_min : float
        m, Inboard VV inner radius
    VV 2nd shell R_max : float
        m, Inboard VV inner shell outer radius
    WC VV shield R_min : float
        m, Inter-VV tungsten carbide neutron shield inner radius
    WC VV shield R_max : float
        m, Inter-VV tungsten carbide neutron shield outer radius
    VV 1st shell R_min : float
        m, Inner radius of the outer VV shell
    VV R_max : float
        m, Vacuum vessel outer radius
    WC shield R_min : float
        m, WC shield inner radius
    WC shield R_max : float
        m, WC shield outer radius
    blanket R_min : float
        m, Blanket inner radius
    blanket R_max : float
        m, Blanket outer radius
    FW R_min : float
        m, First wall inner radius
    FW R_max : float
        m, First wall outer radius
    plasma R_min : float
        m, Inner radius of plasma at midplane
    R0 : float
    plasma R_max : float
        m, Outer radius of plasma at midplane

    A : float
        Plasma aspect ratio
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['radial_build', 'inboard'])
            self.cs_tf_gap = ac(["oh-tf gap thickness"], units="m")
            self.tf_tpt = ac(["tf tpt"], units="m")
            self.vv_tpt = ac(["vv tpt"], units="m")
            self.wedge_assy_fitup_thickness = ac(
                ["wedge assy fit-up thickness"], units="m")
            self.thermal_shield_thickness = ac(
                ["thermal shield insulation thickness"], units="m")
            self.vv_tf_gap = ac(["vv tf gap thickness"], units="m")
            self.vv_shell_thickness = ac(["vv shell thickness"], units="m")
            self.vv_tpt = ac(["vv tpt"], units="m")

        self.add_input("CS R_max", units="m", desc="CS outer radius")

        self.add_input("TF R_max",
                       units='m',
                       desc="Inboard TF leg casing maximum radius")
        self.add_input(
            "WC VV shield thickness",
            units="m",
            desc="Tungsten-carbide neutron shield b/w vacuum vessel shells")
        self.add_input("WC shield thickness",
                       units="m",
                       desc="Tungsten-carbide neutron shield thickness")
        self.add_input("blanket thickness",
                       units="m",
                       desc="Inboard blanket thickness")
        self.add_input("FW thickness", units="m", desc="first wall thickness")
        self.add_input("SOL width", units="m", desc="scrape-off-layer width")
        self.add_input("a", units="m", desc="Plasma minor radius")

        self.add_output("TF R_min", units="m",
                        desc="Inboard TF leg inner radius")
        # vacuum vessel inner extent
        self.add_output("VV R_min", units='m', desc="Inboard VV inner radius")
        self.add_output("VV 2nd shell R_max",
                        units='m',
                        desc="Inboard VV inner shell outer radius")
        # second tungsten carbide shield, within the two VV halves
        self.add_output("WC VV shield R_min",
                        units='m',
                        desc="WC in-VV shield inner radius")
        self.add_output("WC VV shield R_max",
                        units='m',
                        desc="WC in-VV shield outer radius")
        # vacuum vessel
        self.add_output("VV 1st shell R_min",
                        units='m',
                        desc="VV outer shell inner radius")
        self.add_output("VV R_max", units='m', desc="VV outer radius")

        # first tungsten carbide shield
        self.add_output("WC shield R_min",
                        units='m',
                        desc="WC shield inner radius")
        self.add_output("WC shield R_max",
                        units='m',
                        desc="WC shield outer radius")

        self.add_output("blanket R_min",
                        units='m',
                        desc="blanket minimum radius")
        self.add_output("blanket R_max",
                        units='m',
                        desc="blanket maximum radius")

        self.add_output("FW R_min", units='m', desc="first wall inner radius")
        self.add_output("FW R_max", units='m', desc="first wall outer radius")
        self.add_output("plasma R_min", ref=3,
                        units="m",
                        desc="Inner radius of plasma at midplane")
        self.add_output("R0", units="m", ref=4, desc="Plasma major radius")
        self.add_output("plasma R_max", units="m", ref=5,
                        desc="Outer radius of plasma at midplane")
        self.add_output("A", desc="Plasma aspect ratio", ref=3)

    def setup_partials(self):
        outs = [
            "VV R_min", "VV 2nd shell R_max",
            "WC VV shield R_min", "WC VV shield R_max", "VV 1st shell R_min",
            "VV R_max", "WC shield R_min", "WC shield R_max",
            "blanket R_min", "blanket R_max", "FW R_min", "FW R_max",
            "plasma R_min", "R0",
        ]
        self.declare_partials(outs[:], "TF R_max", val=1)
        self.declare_partials(outs[3:], "WC VV shield thickness", val=1)
        self.declare_partials(outs[7:], "WC shield thickness", val=1)
        self.declare_partials(outs[9:], "blanket thickness", val=1)
        self.declare_partials(outs[11:], "FW thickness", val=1)
        self.declare_partials(outs[12:], "SOL width", val=1)
        self.declare_partials(outs[13:], "a", val=1)
        ins = ["TF R_max", "WC VV shield thickness",
               "WC shield thickness", "blanket thickness",
               "FW thickness", "SOL width", "a"]
        self.declare_partials("plasma R_max", ins[:-1], val=1)
        self.declare_partials("plasma R_max", ins[-1], val=2)
        self.declare_partials("A", ins)

        self.declare_partials("TF R_min", "CS R_max", val=1)

    def compute(self, inputs, outputs):
        outputs["TF R_min"] = inputs["CS R_max"] + self.cs_tf_gap

        tf_r_max = inputs["TF R_max"]
        vv_r_min = tf_r_max + self.tf_tpt + self.vv_tpt + \
            self.wedge_assy_fitup_thickness + \
            self.thermal_shield_thickness + self.vv_tf_gap
        outputs["VV R_min"] = vv_r_min
        wc_vv_shield_r_min = vv_r_min + self.vv_shell_thickness
        outputs["VV 2nd shell R_max"] = wc_vv_shield_r_min
        outputs["WC VV shield R_min"] = wc_vv_shield_r_min  # same
        wc_vv_ΔR = inputs["WC VV shield thickness"]
        wc_vv_shield_r_max = wc_vv_shield_r_min + wc_vv_ΔR
        outputs["WC VV shield R_max"] = wc_vv_shield_r_max
        outputs["VV 1st shell R_min"] = wc_vv_shield_r_max  # same
        vv_r_max = wc_vv_shield_r_max + self.vv_shell_thickness
        outputs["VV R_max"] = vv_r_max
        outputs["WC shield R_min"] = vv_r_max  # same

        wc_shield_ΔR = inputs["WC shield thickness"]
        wc_r_max = vv_r_max + wc_shield_ΔR
        outputs["WC shield R_max"] = wc_r_max
        bb_r_min = wc_r_max + self.vv_tpt
        outputs["blanket R_min"] = bb_r_min
        bb_r_max = bb_r_min + inputs["blanket thickness"]
        outputs["blanket R_max"] = bb_r_max
        outputs["FW R_min"] = bb_r_max  # same
        fw_r_max = bb_r_max + inputs["FW thickness"]
        outputs["FW R_max"] = fw_r_max
        plasma_r_min = fw_r_max + inputs["SOL width"]
        outputs["plasma R_min"] = plasma_r_min

        R0 = plasma_r_min + inputs["a"]
        outputs["R0"] = R0
        outputs["plasma R_max"] = plasma_r_min + 2 * inputs["a"]
        outputs["A"] = R0 / inputs["a"]

    def compute_partials(self, inputs, J):
        a = inputs["a"]
        J["A", "TF R_max"] = 1/a
        J["A", "WC VV shield thickness"] = 1/a
        J["A", "WC shield thickness"] = 1/a
        J["A", "blanket thickness"] = 1/a
        J["A", "FW thickness"] = 1/a
        J["A", "SOL width"] = 1/a

        # essentially, recompute R0
        tf_r_max = inputs["TF R_max"]
        vv_r_min = tf_r_max + self.tf_tpt + self.vv_tpt + \
            self.wedge_assy_fitup_thickness + \
            self.thermal_shield_thickness + self.vv_tf_gap
        wc_vv_shield_r_min = vv_r_min + self.vv_shell_thickness
        wc_vv_ΔR = inputs["WC VV shield thickness"]
        wc_vv_shield_r_max = wc_vv_shield_r_min + wc_vv_ΔR
        vv_r_max = wc_vv_shield_r_max + self.vv_shell_thickness
        wc_shield_ΔR = inputs["WC shield thickness"]
        wc_r_max = vv_r_max + wc_shield_ΔR
        bb_r_min = wc_r_max + self.vv_tpt
        bb_r_max = bb_r_min + inputs["blanket thickness"]
        fw_r_max = bb_r_max + inputs["FW thickness"]
        plasma_r_min = fw_r_max + inputs["SOL width"]
        # - (R0 - a) / a**2
        J["A", "a"] = - (plasma_r_min)/a**2


class MenardSTOutboardRadialBuild(om.ExplicitComponent):
    r"""Outboard radial build

    Inputs
    ------
    plasma R_max : float
        m, Outer radius of the core plasma at midplane
    SOL width : float
        m, low field side scrape-off-layer width
    blanket thickness : float
        m, Thickness of FW and blanket
    access thickness : float
        m, thickness of access region
    VV thickness : float
        m, thickness of vacuum vessel
    shield thickness : float
        m, thickness of neutron shield

    Outputs
    -------
    blanket R_in : float
        m, inner radius of outboard blanket at midplane
    blanket R_out : float
        m, outer radius of outboard blanket at midplane
    TF R_min : float
        m, minimum radius of inner part of TF outboard leg
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']

        # outboard build
        self.add_input("plasma R_max",
                       units="m",
                       desc="outer radius of core at midplane")
        self.add_input("SOL width",
                       units="m",
                       desc="low field side scrape-off-layer width")
        self.add_input("blanket thickness", units="m")

        # I think this is where the PF coils go?
        self.add_input("access thickness", units="m", desc="Access thickness")
        self.add_input("VV thickness", units="m", desc="Vacuum vessel")
        self.add_input("shield thickness", units="m", desc="Neutron shield")

        self.add_output("blanket R_in", units='m', lower=0)
        self.add_output("blanket R_out", units='m', lower=0)
        self.add_output("TF R_min",
                        units='m',
                        desc="TF leg casing minimum radius")

    def setup_partials(self):
        self.declare_partials("TF R_min", "plasma R_max", val=1)
        self.declare_partials("TF R_min", "SOL width", val=1)
        self.declare_partials("TF R_min", "blanket thickness", val=1)
        self.declare_partials("TF R_min", "access thickness", val=1)
        self.declare_partials("TF R_min", "VV thickness", val=1)
        self.declare_partials("TF R_min", "shield thickness", val=1)

        self.declare_partials("blanket R_in", ["plasma R_max", "SOL width"],
                              val=1)
        self.declare_partials("blanket R_out", ["plasma R_max", "SOL width",
                                                "blanket thickness"], val=1)

    def compute(self, inputs, outputs):
        # outboard build
        plasma_r_max = inputs["plasma R_max"]
        lfs_sol_width = inputs["SOL width"]
        ob_blanket_thickness = inputs["blanket thickness"]
        ob_access_thickness = inputs["access thickness"]
        ob_vv_thickness = inputs["VV thickness"]
        ob_shield_thickness = inputs["shield thickness"]

        outputs["blanket R_in"] = plasma_r_max + lfs_sol_width
        outputs["blanket R_out"] = (
            outputs["blanket R_in"] + ob_blanket_thickness)

        ob_build = [
            plasma_r_max, lfs_sol_width, ob_blanket_thickness,
            ob_access_thickness, ob_vv_thickness, ob_shield_thickness
        ]
        outputs["TF R_min"] = sum(ob_build)


class MenardSTOuterMachineRadialBuild(om.ExplicitComponent):
    r"""Radial build outside the TF coils

    Inputs
    ------
    Ob TF R_out : float
        m, Outboard radius of the outboard leg
    TF-cryostat thickness : float
        m, Outboard TF outer radius to cryostat outer wall

    Outputs
    -------
    cryostat R_out: float
        m, Outer radius of the cryostat
    """

    def setup(self):
        self.add_input("Ob TF R_out",
                       units="m",
                       desc="Outboard radius of the outboard leg")
        self.add_input("TF-cryostat thickness",
                       units="m",
                       desc="Outboard TF outer radius to cryostat outer wall")
        self.add_output("cryostat R_out",
                        units="m",
                        desc="Outer radius of the cryostat")

    def compute(self, inputs, outputs):
        TF_R_out = inputs["Ob TF R_out"]
        TF_to_cryostat_thickness = inputs["TF-cryostat thickness"]

        outputs["cryostat R_out"] = TF_R_out + TF_to_cryostat_thickness

    def setup_partials(self):
        self.declare_partials("cryostat R_out", "Ob TF R_out", val=1)
        self.declare_partials("cryostat R_out", "TF-cryostat thickness", val=1)


class STRadialBuild(om.Group):
    r"""The high-level radial build for the device.

    The model is broken into a few parts.
    The Inboard subsystem builds the device outward from the center.
    The first input is 'CS R_max', and it yields 'TF R_min'.

    The TF thickness is *not* an input to this component, since it will be
    decided independently by the Inboard TF component. This is important for
    ensuring positivity.

    The second key input is 'TF R_max'. This forms the base of the 'stack' of
    components outside the TF: the vacuum vessel, shields, blanket, first wall,
    SOL, and the plasma itself. The 'inboard' stack actually goes out to
    'plasma R_max', the outer radius of the plasma at midplane.

    The Outboard subsystem 'ob' is less complex. It builds from 'plasma R_max'
    to the 'Ob TF R_min', the minimum radius for the outboard TF.
    Note that this may not be the actual TF radius of the start of the TF;
    it may be larger for certain coil shapes.

    The Outer Machine subsystem 'ob', builds the machine outside the
    outer leg of the TF, to the cryostat.

    Below are listed particularly interesting inputs and outputs.

    Inputs
    ------
    CS R_max : float
        m, CS outer radius

    Ib TF R_max : float
        m, Inboard TF leg outer radius
    a : float
        m, Plasma minor radius

    Ob TF R_out : float
        m, Outboard radius of the outboard leg

    Additional inputs to the 'ib', 'ob', and 'om' components are provided
    by the configuration file :code:`radial_build.yaml`.

    Outputs
    -------
    Ib TF R_min : float
        m, Inboard TF leg inner radius
    R0 : float
    plasma R_max : float
        m, Outer radius of plasma at midplane

    A : float
        Plasma aspect ratio

    Ob TF R_min : float
        m, Outboard TF leg minimum inner radius
    cryostat R_out : float
        m, Outer radius of the cryostat
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']

        self.add_subsystem('props', RadialBuildProperties(config=config))
        self.add_subsystem('ib',
                           MenardSTInboardRadialBuild(config=config),
                           promotes_inputs=["CS R_max", "a",
                                            ("TF R_max", "Ib TF R_max")],
                           promotes_outputs=[("TF R_min", "Ib TF R_min"),
                                             "plasma R_max",
                                             "plasma R_min",
                                             "R0", "A"])
        self.add_subsystem('ob',
                           MenardSTOutboardRadialBuild(config=config),
                           promotes_inputs=["plasma R_max"],
                           promotes_outputs=[("TF R_min", "Ob TF R_min")])

        # to be computed after TF thickness is determined
        self.add_subsystem('om',
                           MenardSTOuterMachineRadialBuild(),
                           promotes_inputs=["Ob TF R_out"],
                           promotes_outputs=["cryostat R_out"])

        # connect inputs
        self.connect('props.Ib WC VV shield thickness',
                     ['ib.WC VV shield thickness'])
        self.connect('props.Ib FW thickness', ['ib.FW thickness'])
        self.connect('props.Ib SOL width', ['ib.SOL width'])

        self.connect('props.Ob SOL width', ['ob.SOL width'])
        self.connect('props.Ob blanket thickness', ['ob.blanket thickness'])
        self.connect('props.Ob access thickness', ['ob.access thickness'])
        self.connect('props.Ob vv thickness', ['ob.VV thickness'])
        self.connect('props.Ob shield thickness', ['ob.shield thickness'])

        self.connect('props.TF-cryostat thickness',
                     ['om.TF-cryostat thickness'])


class MenardSTRadialBuild(om.Group):
    r"""The high-level radial build for the device.

    In this model, the inboard blanket and shield thicknesses are set as a
    function of aspect ratio.
    This is done to align with Menard's spreadsheet model.

    A nonlinear solver needs to be attached to this component.
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']

        self.add_subsystem('ST_radial_build',
                           STRadialBuild(config=config),
                           promotes_inputs=["CS R_max",
                                            "Ib TF R_max",
                                            "a",
                                            "Ob TF R_out"],
                           promotes_outputs=["*"])

        self.add_subsystem("ib_blanket",
                           MenardInboardBlanketFit(config=config),
                           promotes_inputs=["A"])
        self.add_subsystem("ib_shield",
                           MenardInboardShieldFit(config=config),
                           promotes_inputs=["A"])

        self.connect('ib_shield.shield_thickness', [
                     'ST_radial_build.ib.WC shield thickness'])
        self.connect('ib_blanket.blanket_thickness', [
                     'ST_radial_build.ib.blanket thickness'])


if __name__ == "__main__":

    # test building a tokamak with A=1.6
    class MyRadialBuild(om.Group):

        def initialize(self):
            self.options.declare('config', default=None)

        def setup(self):
            config = self.options['config']
            self.add_subsystem("mrb", MenardSTRadialBuild(config=config))

            # balance the desired R0 and the actual R0 by changing a
            Rdes = om.ExecComp("Rdes = 3.0", Rdes={"units": "m"})
            Rbal = om.BalanceComp()
            Rbal.add_balance('a', normalize=True, units="m", eq_units="m")
            self.add_subsystem("Rdes", Rdes)
            self.add_subsystem("Rbalance", subsys=Rbal)
            self.connect("Rbalance.a", "mrb.a")
            self.connect("Rdes.Rdes", "Rbalance.lhs:a")
            self.connect("mrb.R0", "Rbalance.rhs:a")

    prob = om.Problem()
    uc = UserConfigurator()

    prob.model = MyRadialBuild(config=uc)
    prob.setup()
    prob.set_val("mrb.Ib TF R_max", 0.405)
    prob.set_val("mrb.CS R_max", 0.105)

    model = prob.model
    newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    model.linear_solver = om.DirectSolver()

    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True, units=True)
    all_outputs = prob.model.list_outputs(values=True, units=True)
