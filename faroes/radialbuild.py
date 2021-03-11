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
                       "SOL width",
                       component_name="Ib SOL width",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "fw thickness",
                       component_name="Ib FW thickness",
                       units="m")
        acc.set_output(ivc,
                       f,
                       "vv shielding thickness",
                       component_name="Ib WC VV shield thickness",
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


class MenardSTRadialBuild(om.Group):
    r"""
    New design variables from this group:
    Central solenoid outer diameter
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']

        self.add_subsystem('props', RadialBuildProperties(config=config))
        self.add_subsystem("ib_blanket",
                           MenardInboardBlanketFit(config=config),
                           promotes_inputs=["A"])
        self.add_subsystem("ib_shield",
                           MenardInboardShieldFit(config=config),
                           promotes_inputs=["A"])
        self.add_subsystem('ib',
                           MenardSTInboardRadialBuild(config=config),
                           promotes_inputs=["plasma R_min", "CS R_max"],
                           promotes_outputs=[("TF R_min", "Ib TF R_min"),
                                             ("TF R_max", "Ib TF R_max")])
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
        self.connect('props.Ib SOL width', ['ib.SOL width'])
        self.connect('props.Ib FW thickness', ['ib.FW thickness'])
        self.connect('ib_blanket.blanket_thickness', ['ib.blanket thickness'])
        self.connect('ib_shield.shield_thickness', ['ib.WC shield thickness'])
        self.connect('props.Ib WC VV shield thickness',
                     ['ib.WC VV shield thickness'])

        self.connect('props.Ob SOL width', ['ob.SOL width'])
        self.connect('props.Ob blanket thickness', ['ob.blanket thickness'])
        self.connect('props.Ob access thickness', ['ob.access thickness'])
        self.connect('props.Ob vv thickness', ['ob.VV thickness'])
        self.connect('props.Ob shield thickness', ['ob.shield thickness'])

        self.connect('props.TF-cryostat thickness',
                     ['om.TF-cryostat thickness'])


class MenardSTInboardRadialBuild(om.ExplicitComponent):
    r"""Radial build

    Inputs
    ------
    A : float
        Plasma aspect ratio
    plasma R_min : float
        m, Inner radius of plasma at midplane
    SOL width : float
        m, Scrape-off-layer width
    FW thickness : float
        m, First wall thickness
    blanket thickness : float
        m, Inboard blanket thickness
    WC shield thickness : float
        m, Tungsten-carbide neutron shield thickness
    WC VV shield thickness : float
        m, Tungsten-carbide neutron shield b/w vacuum vessel shells
    CS R_max : float
        m, Central solenoid maximum radius

    Outputs
    -------
    FW R_max : float
        m, First wall outer radius
    FW R_min : float
        m, First wall inner radius
    blanket R_max : float
        m, Blanket outer radius
    blanket R_min : float
        m, Blanket inner radius
    WC shield R_max : float
        m, WC shield outer radius
    WC shield R_min : float
        m, WC shield inner radius
    VV R_max : float
        m, Vacuum vessel outer radius
    VV 1st shell R_min : float
        m, Inner radius of the outer VV shell
    WC VV shield R_max : float
        m, Inter-VV tungsten carbide neutron shield outer radius
    WC VV shield R_min : float
        m, Inter-VV tungsten carbide neutron shield inner radius
    VV 2nd shell R_max : float
        m, Inboard VV inner shell outer radius
    VV R_min : float
        m, Inboard VV inner radius
    TF R_max : float
        m, Inboard TF leg maximum radius
    TF R_min : float
        m, Inboard TF leg minimum radius
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['radial_build', 'inboard'])
            self.vv_tpt = ac(["vv tpt"], units="m")
            self.vv_shell_thickness = ac(["vv shell thickness"], units="m")

            self.vv_tf_gap = ac(["vv tf gap thickness"], units="m")

            self.thermal_shield_thickness = ac(
                ["thermal shield insulation thickness"], units="m")
            self.wedge_assy_fitup_thickness = ac(
                ["wedge assy fit-up thickness"], units="m")
            self.vv_tpt = ac(["vv tpt"], units="m")
            self.tf_tpt = ac(["tf tpt"], units="m")
            self.cs_tf_gap = ac(["oh-tf gap thickness"], units="m")

        self.add_input("plasma R_min",
                       units="m",
                       desc="Inner radius of plasma at midplane")
        self.add_input("SOL width", units="m", desc="scrape-off-layer width")
        self.add_input("FW thickness", units="m", desc="first wall thickness")
        self.add_input("blanket thickness",
                       units="m",
                       desc="Inboard blanket thickness")
        self.add_input("WC shield thickness",
                       units="m",
                       desc="Tungsten-carbide neutron shield thickness")
        self.add_input(
            "WC VV shield thickness",
            units="m",
            desc="Tungsten-carbide neutron shield b/w vacuum vessel shells")

        self.add_input("CS R_max",
                       units="m",
                       desc="Central solenoid max radius")

        self.add_output("FW R_max", units='m', desc="first wall outer radius")
        self.add_output("FW R_min", units='m', desc="first wall inner radius")
        self.add_output("blanket R_max",
                        units='m',
                        desc="blanket maximum radius")
        self.add_output("blanket R_min",
                        units='m',
                        desc="blanket minimum radius")

        # first tungsten carbide shield
        self.add_output("WC shield R_max",
                        units='m',
                        desc="WC shield outer radius")
        self.add_output("WC shield R_min",
                        units='m',
                        desc="WC shield inner radius")

        # vacuum vessel
        self.add_output("VV R_max", units='m', desc="VV outer radius")
        self.add_output("VV 1st shell R_min",
                        units='m',
                        desc="VV outer shell inner radius")

        # second tungsten carbide shield, within the two VV halves
        self.add_output("WC VV shield R_max",
                        units='m',
                        desc="WC in-VV shield outer radius")
        self.add_output("WC VV shield R_min",
                        units='m',
                        desc="WC in-VV shield inner radius")

        # vacuum vessel inner extent
        self.add_output("VV 2nd shell R_max",
                        units='m',
                        desc="Inboard VV inner shell outer radius")
        self.add_output("VV R_min", units='m', desc="Inboard VV inner radius")

        self.add_output("TF R_max",
                        units='m',
                        desc="Inboard TF leg casing maximum radius")
        self.add_output("TF R_min",
                        units='m',
                        desc="Inboard TF leg casing minimum radius")

    def setup_partials(self):
        outs = [
            "FW R_max", "FW R_min", "blanket R_max", "blanket R_min",
            "WC shield R_max", "WC shield R_min", "VV R_max",
            "VV 1st shell R_min", "WC VV shield R_max", "WC VV shield R_min",
            "VV 2nd shell R_max", "VV R_min", "TF R_max"
        ]
        self.declare_partials(outs[:], "plasma R_min", val=1)
        self.declare_partials(outs[:], "SOL width", val=-1)
        self.declare_partials(outs[1:], "FW thickness", val=-1)
        self.declare_partials(outs[3:], "blanket thickness", val=-1)
        self.declare_partials(outs[5:], "WC shield thickness", val=-1)
        self.declare_partials(outs[9:], "WC VV shield thickness", val=-1)

        self.declare_partials("TF R_min", ["CS R_max"], val=1)

    def compute(self, inputs, outputs):
        # inboard build, from the plasma inward
        fw_r_max = inputs["plasma R_min"] - inputs["SOL width"]
        outputs["FW R_max"] = fw_r_max
        fw_r_min = fw_r_max - inputs["FW thickness"]
        outputs["FW R_min"] = fw_r_min
        outputs["blanket R_max"] = fw_r_min  # they are the same
        bb_r_min = fw_r_min - inputs["blanket thickness"]
        outputs["blanket R_min"] = bb_r_min

        wc_r_max = bb_r_min - self.vv_tpt
        outputs["WC shield R_max"] = wc_r_max
        wc_r_min = wc_r_max - inputs["WC shield thickness"]
        outputs["WC shield R_min"] = wc_r_min

        outputs["VV R_max"] = wc_r_min  # same
        vv_1_r_min = wc_r_min - self.vv_shell_thickness
        outputs["VV 1st shell R_min"] = vv_1_r_min
        outputs["WC VV shield R_max"] = vv_1_r_min  # same
        wc_vv_shield_r_min = vv_1_r_min - inputs["WC VV shield thickness"]
        outputs["WC VV shield R_min"] = wc_vv_shield_r_min
        outputs["VV 2nd shell R_max"] = wc_vv_shield_r_min  # same
        vv_r_min = wc_vv_shield_r_min - self.vv_shell_thickness
        outputs["VV R_min"] = vv_r_min
        tf_r_max = vv_r_min - self.vv_tf_gap \
            - self.thermal_shield_thickness - self.wedge_assy_fitup_thickness \
            - self.vv_tpt - self.tf_tpt
        outputs["TF R_max"] = tf_r_max

        # start from the inside
        outputs["TF R_min"] = inputs["CS R_max"] + self.cs_tf_gap


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


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()

    prob.model = MenardSTRadialBuild(config=uc)
    prob.setup()
    prob.set_val("A", 2.6)
    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True, units=True)
    all_outputs = prob.model.list_outputs(values=True, units=True)
