import openmdao.api as om
from faroes.configurator import UserConfigurator, Accessor

from faroes.simple_tf_magnet import InboardMagnetGeometry
from faroes.simple_tf_magnet import OutboardMagnetGeometry

from faroes.blanket import MenardInboardBlanketFit
from faroes.blanket import MenardInboardShieldFit
from faroes.blanket import OutboardBlanketFit


class Properties(om.Group):
    r"""Helper for the Menard ST radial build

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Outputs
    -------
    Ib WC VV shield thickness : float
        m
    Ib FW thickness : float
        m
    Ib SOL width : float
        m
    Ob SOL width : float
        m
    Ob access thickness : float
        m
    Ob vv thickness : float
        m
    TF-cryostat thickness : float
        m
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

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


class CSToTF(om.ExplicitComponent):
    r"""Radial build out to the inside of the TF

    A diagram of the radial build is shown below. On the left, (after the first
    entry) are the thicknesses of the various components. On the right are the
    absolute radii of various points (generally the inner or outer extent of
    components). Two entries on the same line have equal radii.::

             Inputs   Outputs

         Plug R_out + CS R_in
              CS ΔR |
                    + CS R_out
          CS-TF gap |
                    + TF R_in

    Notes
    -----

    "CS R_in" is equal to "Plug R_out".

    Options
    -------
    config : UserConfigurator
        Configuration tree. Used to load a default value for "CS-TF gap", else
        it is zero.

    Inputs
    ------
    Plug R_out : float
        m, outer radius of a central plug
    CS ΔR : float
        m, thickness of CS
    CS-TF gap: float
        m, Extra distance between the outside of the CS and the inner structure
        of the TF.

    Outputs
    -------
    CS R_in : float
        m, Inner radius of CS
    CS R_out : float
        m, Inner radius of CS
    TF R_in : float
        m, Inboard TF leg inner radius

    Notes
    -----
    Loads a default value for the cs-to-tf gap from the configuration tree.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['radial_build', 'inboard'])
            self.cs_tf_gap = ac(["oh-tf gap thickness"], units="m")
        else:
            self.cs_tf_gap = 0

        self.add_input("Plug R_out",
                       units='m',
                       val=0,
                       desc="Central plug outer radius")
        self.add_input("CS ΔR",
                       units='m',
                       val=0,
                       desc="Central solenoid radial width")
        self.add_input("CS-TF gap",
                       units='m',
                       val=self.cs_tf_gap,
                       desc="Gap between CS and TF magnets")
        self.add_output("CS R_in",
                        units='m',
                        lower=0,
                        desc="Central solenoid inner radius")
        self.add_output("CS R_out",
                        units='m',
                        lower=0,
                        desc="Central solenoid outer radius")
        self.add_output("TF R_in",
                        units="m",
                        desc="Inboard TF leg inner radius",
                        lower=0)

    def compute(self, inputs, outputs):
        plug_r_out = inputs["Plug R_out"]
        cs_r_in = plug_r_out
        outputs["CS R_in"] = cs_r_in
        cs_Δr = inputs["CS ΔR"]
        cs_r_out = cs_r_in + cs_Δr
        outputs["CS R_out"] = cs_r_out
        cs_tf_gap = inputs["CS-TF gap"]
        outputs["TF R_in"] = cs_r_out + cs_tf_gap

    def setup_partials(self):
        self.declare_partials("CS R_in", "Plug R_out", val=1)
        self.declare_partials(["CS R_out", "TF R_in"], ["CS ΔR", "Plug R_out"],
                              val=1)
        self.declare_partials(["TF R_in"], ["CS-TF gap"], val=1)


class MenardSTInboard(om.ExplicitComponent):
    r"""From outside of the TF to the first wall

    A diagram of the radial build is shown below. On the left, (after the first
    entry) are the thicknesses of the various components. On the right are the
    absolute radii of various points (generally the inner or outer extent of
    components). Two entries on the same line have equal radii.::

                                Inputs   Outputs

                              TF R_out +
           *TF true position tolerance |
                    *vacuum vessel TPT |
      *wedge assembly fit-up thickness |
                                       + Thermal shield R_in
             *Thermal shield thickness |
                            *vv ts gap |
                                       + VV R_in
             *vv inner shell thickness |
                                       + VV 2nd shell R_out, WC VV shield R_in
                WC VV shield thickness |
                                       + WC VV shield R_out, VV 1st shell R_in
             *vv outer shell thickness |
                                       + VV R_out, WC shield R_in
                   WC shield thickness |
                                       + WC shield R_out
          *extra shield to blanket gap |
                    *vacuum vessel TPT |
                                       + blanket R_in
                     blanket thickness |
                                       + blanket R_out, FW R_in
                          FW thickness |
                                       + FW R_out

    The components on the left marked with a * are loaded from the
    configurator.

    Options
    -------
    config : UserConfigurator
        Configuration tree.

    Inputs
    ------
    TF R_out : float
        m, Inboard TF leg outer radius
    WC VV shield thickness : float
        m, Tungsten-carbide neutron shield b/w vacuum vessel shells
    WC shield thickness : float
        m, Tungsten-carbide neutron shield thickness
    blanket thickness : float
        m, Inboard blanket thickness
    FW thickness : float
        m, First wall thickness

    Outputs
    -------
    Thermal shield R_in : float
        m, Inner radius of thermal shield
    VV R_in : float
        m, Inboard VV inner radius
    VV 2nd shell R_out : float
        m, Inboard VV inner shell outer radius
    WC VV shield R_in : float
        m, Inter-VV tungsten carbide neutron shield inner radius
    WC VV shield R_out : float
        m, Inter-VV tungsten carbide neutron shield outer radius
    VV 1st shell R_in : float
        m, Inner radius of the outer VV shell
    VV R_out : float
        m, Vacuum vessel outer radius
    WC shield R_in : float
        m, WC shield inner radius
    WC shield R_out : float
        m, WC shield outer radius
    blanket R_in : float
        m, Blanket inner radius
    blanket R_out : float
        m, Blanket outer radius
    FW R_in : float
        m, First wall inner radius
    FW R_out : float
        m, First wall outer radius

    Thermal shield to FW : float
        m, Total thickness of region including
        the thermal shield and inboard FW
    Shield to FW: float
        m, Total thickness from the outer WC shield to FW
    Blanket to FW : float
        m, Total thickness from the blanket to FW.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['radial_build', 'inboard'])
            self.tf_tpt = ac(["tf tpt"], units="m")
            self.vv_tpt = ac(["vv tpt"], units="m")
            self.wedge_assy_fitup_thickness = ac(
                ["wedge assy fit-up thickness"], units="m")
            self.thermal_shield_thickness = ac(
                ["thermal shield insulation thickness"], units="m")
            vv_ts_gap = ac(["vv ts gap thickness"], units="m")
            self.vv_inner_shell_thickness = ac(["vv inner shell thickness"],
                                               units="m")
            self.vv_outer_shell_thickness = ac(["vv outer shell thickness"],
                                               units="m")
            extra_shield_to_blanket_gap = ac(["extra shield to blanket gap"],
                                             units="m")

        self.add_input("TF R_out",
                       units='m',
                       desc="Inboard TF leg casing outer radius")
        self.add_input("Thermal shield to VV gap",
                       units='m',
                       val=vv_ts_gap,
                       desc="Gap between thermal shield and vacuum vessel")
        self.add_input(
            "WC VV shield thickness",
            units="m",
            desc="Tungsten-carbide neutron shield b/w vacuum vessel shells")
        self.add_input("WC shield thickness",
                       units="m",
                       desc="Tungsten-carbide neutron shield thickness")
        self.add_input("Extra shield to blanket gap",
                       units="m",
                       val=extra_shield_to_blanket_gap,
                       desc="Extra gap between shield & blanket")
        self.add_input("blanket thickness",
                       units="m",
                       desc="Inboard blanket thickness")
        self.add_input("FW thickness", units="m", desc="First wall thickness")

        # Thermal shield inner radius
        self.add_output("Thermal shield R_in",
                        units="m",
                        desc="Thermal shield inner radius")
        # vacuum vessel inner extent
        self.add_output("VV R_in", units='m', desc="Inboard VV inner radius")
        self.add_output("VV 2nd shell R_out",
                        units='m',
                        desc="Inboard VV inner shell outer radius")
        # second tungsten carbide shield, within the two VV halves
        self.add_output("WC VV shield R_in",
                        units='m',
                        desc="WC in-VV shield inner radius")
        self.add_output("WC VV shield R_out",
                        units='m',
                        desc="WC in-VV shield outer radius")
        # vacuum vessel
        self.add_output("VV 1st shell R_in",
                        units='m',
                        desc="VV outer shell inner radius")
        self.add_output("VV R_out", units='m', desc="VV outer radius")

        # first tungsten carbide shield
        self.add_output("WC shield R_in",
                        units='m',
                        desc="WC shield inner radius")
        self.add_output("WC shield R_out",
                        units='m',
                        desc="WC shield outer radius")

        self.add_output("blanket R_in", units='m', desc="Blanket inner radius")
        self.add_output("blanket R_out",
                        units='m',
                        desc="Blanket outer radius")

        self.add_output("FW R_in", units='m', desc="First wall inner radius")
        self.add_output("FW R_out", units='m', desc="First wall outer radius")

        self.add_output("Thermal shield to FW",
                        units="m",
                        desc="From thermal shield inner to fw outer")
        self.add_output("Shield to FW",
                        units="m",
                        desc="From shield inner to FW outer")
        self.add_output("Blanket to FW",
                        units="m",
                        desc="From blanket inner to FW outer")

    def setup_partials(self):
        outs = [
            "Thermal shield R_in",
            "VV R_in",
            "VV 2nd shell R_out",
            "WC VV shield R_in",
            "WC VV shield R_out",
            "VV 1st shell R_in",
            "VV R_out",
            "WC shield R_in",
            "WC shield R_out",
            "blanket R_in",
            "blanket R_out",
            "FW R_in",
            "FW R_out",
        ]
        self.declare_partials(outs[:], "TF R_out", val=1)
        self.declare_partials(outs[1:], "Thermal shield to VV gap", val=1)
        self.declare_partials(outs[4:], "WC VV shield thickness", val=1)
        self.declare_partials(outs[8:], "WC shield thickness", val=1)
        self.declare_partials(outs[9:], "Extra shield to blanket gap", val=1)
        self.declare_partials(outs[10:], "blanket thickness", val=1)
        self.declare_partials(outs[12:], "FW thickness", val=1)

        self.declare_partials("Thermal shield to FW", [
            "Thermal shield to VV gap", "WC VV shield thickness",
            "WC shield thickness", "Extra shield to blanket gap",
            "blanket thickness", "FW thickness"
        ],
                              val=1)
        self.declare_partials("Shield to FW", [
            "WC shield thickness", "Extra shield to blanket gap",
            "blanket thickness", "FW thickness"
        ],
                              val=1)
        self.declare_partials("Blanket to FW",
                              ["blanket thickness", "FW thickness"],
                              val=1)

    def compute(self, inputs, outputs):

        tf_r_out = inputs["TF R_out"]
        thermal_shield_r_in = tf_r_out + self.tf_tpt + \
            self.vv_tpt + self.wedge_assy_fitup_thickness

        ts_vv_gap = inputs["Thermal shield to VV gap"]
        outputs["Thermal shield R_in"] = thermal_shield_r_in
        vv_r_in = thermal_shield_r_in + self.thermal_shield_thickness + \
            ts_vv_gap
        outputs["VV R_in"] = vv_r_in
        wc_vv_shield_r_in = vv_r_in + self.vv_inner_shell_thickness
        outputs["VV 2nd shell R_out"] = wc_vv_shield_r_in
        outputs["WC VV shield R_in"] = wc_vv_shield_r_in  # same
        wc_vv_ΔR = inputs["WC VV shield thickness"]
        wc_vv_shield_r_out = wc_vv_shield_r_in + wc_vv_ΔR
        outputs["WC VV shield R_out"] = wc_vv_shield_r_out
        outputs["VV 1st shell R_in"] = wc_vv_shield_r_out  # same
        vv_r_out = wc_vv_shield_r_out + self.vv_outer_shell_thickness
        outputs["VV R_out"] = vv_r_out
        wc_r_in = vv_r_out
        outputs["WC shield R_in"] = wc_r_in  # same

        wc_shield_ΔR = inputs["WC shield thickness"]
        wc_r_out = vv_r_out + wc_shield_ΔR
        outputs["WC shield R_out"] = wc_r_out
        extra_sh_to_bl_gap = inputs["Extra shield to blanket gap"]
        bb_r_in = wc_r_out + self.vv_tpt + extra_sh_to_bl_gap
        outputs["blanket R_in"] = bb_r_in
        bb_r_out = bb_r_in + inputs["blanket thickness"]
        outputs["blanket R_out"] = bb_r_out
        outputs["FW R_in"] = bb_r_out  # same
        fw_r_out = bb_r_out + inputs["FW thickness"]
        outputs["FW R_out"] = fw_r_out

        # This totals up part of the radial build
        outputs["Thermal shield to FW"] = fw_r_out - thermal_shield_r_in
        outputs["Shield to FW"] = fw_r_out - wc_r_in
        outputs["Blanket to FW"] = fw_r_out - bb_r_in


class Plasma(om.ExplicitComponent):
    r"""From inboard FW to outboard FW

    A diagram of the radial build is shown below. On the left, (after the first
    entry) are the thicknesses of the various components. On the right are the
    absolute radii of various points.::

            Inputs   Outputs

       Ib FW R_out +
      Ib SOL width |
                   + R_in
                 a |
                   + R0
                 a |
                   + R_out
      Ob SOL width |
                   + Ob FW R_in

    Inputs
    ------
    Ib FW R_out : float
        m, Outer radius of inboard first wall
    Ib SOL width : float
        m, Inboard scrape-off-layer width
    a : float
        m, Plasma minor radius
    Ob SOL width : float
        m, Outboard scrape-off-layer width

    Outputs
    -------
    R_in : float
        m, Inner radius of LCFS at midplane
    R0 : float
        m, Plasma major radius
    R_out : float
        m, Outer radius of LCFS at midplane
    Ob FW R_in : float
        m, Inner radius of the outboard first wall

    A : float
        Plasma aspect ratio

    """
    def setup(self):
        self.add_input("Ib FW R_out",
                       units="m",
                       desc="Inboard first wall outer radius")
        self.add_input("Ib SOL width",
                       units="m",
                       desc="Inboard scrape-off-layer width")
        self.add_input("a", units="m", desc="Plasma minor radius")
        self.add_input("Ob SOL width",
                       units="m",
                       desc="Outboard scrape-off-layer width")

        self.add_output("R_in",
                        ref=3,
                        units="m",
                        desc="Inner radius of plasma at midplane")
        self.add_output("R0",
                        units="m",
                        ref=4,
                        desc="Plasma major radius",
                        lower=0)
        self.add_output("R_out",
                        units="m",
                        ref=5,
                        desc="Outer radius of plasma at midplane",
                        lower=0)
        self.add_output("Ob FW R_in",
                        units="m",
                        desc="Inner radius of the outboard first wall",
                        ref=5)
        self.add_output("A", desc="Plasma aspect ratio", ref=3, lower=0)

    def compute(self, inputs, outputs):
        fw_r_out = inputs["Ib FW R_out"]
        plasma_r_in = fw_r_out + inputs["Ib SOL width"]
        outputs["R_in"] = plasma_r_in
        a = inputs["a"]

        R0 = plasma_r_in + a
        outputs["R0"] = R0
        plasma_r_out = plasma_r_in + 2 * a
        outputs["R_out"] = plasma_r_out
        outputs["Ob FW R_in"] = plasma_r_out + inputs["Ob SOL width"]

        outputs["A"] = R0 / a

    def setup_partials(self):
        self.declare_partials("R_in", ["Ib FW R_out", "Ib SOL width"], val=1)
        self.declare_partials("R0", ["Ib FW R_out", "Ib SOL width", "a"],
                              val=1)
        self.declare_partials("R_out", ["Ib FW R_out", "Ib SOL width"], val=1)
        self.declare_partials("R_out", ["a"], val=2)
        self.declare_partials("Ob FW R_in",
                              ["Ib FW R_out", "Ib SOL width", "Ob SOL width"],
                              val=1)
        self.declare_partials("Ob FW R_in", ["a"], val=2)
        self.declare_partials("A", ["Ib FW R_out", "Ib SOL width", "a"])

    def compute_partials(self, inputs, J):
        fw_r_out = inputs["Ib FW R_out"]
        ib_sol = inputs["Ib SOL width"]
        a = inputs["a"]
        J["A", "Ib FW R_out"] = 1 / a
        J["A", "Ib SOL width"] = 1 / a
        J["A", "a"] = -(fw_r_out + ib_sol) / a**2


class MenardSTOutboard(om.ExplicitComponent):
    r"""Outboard radial build

    A diagram of the radial build is shown below. On the left, (after the first
    entry) are the thicknesses of the various components. On the right are the
    absolute radii of various points (generally the inner or outer extent of
    components). Two entries on the same line have equal radii.::

                 Inputs   Outputs

             Ob FW R_in + blanket R_in
      blanket thickness |
                        + blanket R_out
       shield thickness |
                        + shield R_out
       access thickness |
           VV thickness |
                        + TF R_min
           gap thickness|
                        + TF R_in

    Inputs
    ------
    Ob FW R_in : float
        m, Outboard midplane first wall inner radius
    blanket thickness : float
        m, Thickness of FW and blanket
    shield thickness : float
        m, Thickness of neutron shield
    access thickness : float
        m, Thickness of access region
    VV thickness : float
        m, Thickness of vacuum vessel
    gap thickness : float
        m, Thickness of extra gap in front of TF

    Outputs
    -------
    blanket R_in : float
        m, Inner radius of outboard blanket at midplane
    blanket R_out : float
        m, Outer radius of outboard blanket at midplane
    shield R_out : float
        m, Outer radius of outboard blanket at midplane
    TF R_min : float
        m, Minimum radius of inner part of TF outboard leg
    TF R_in : float
        m, Inner radius of inner part of TF outboard leg
    """
    def setup(self):

        # outboard build
        self.add_input("Ob FW R_in",
                       units="m",
                       desc="Outboard midplane first wall inner radius")
        self.add_input("blanket thickness", units="m", val=1.0,
                       desc="Thickness of first wall & blanket")

        self.add_input("access thickness", units="m", desc="Access thickness")
        self.add_input("VV thickness", units="m", desc="Vacuum vessel")
        self.add_input("shield thickness", units="m", desc="Neutron shield")
        self.add_input("gap thickness",
                       units="m",
                       val=0,
                       desc="Extra gap in front of the TF")

        self.add_output("blanket R_in",
                        units='m',
                        lower=0,
                        desc="Outboard blanket inner radius")
        self.add_output("blanket R_out",
                        units='m',
                        lower=0,
                        desc="Outboard blanket outer radius")

        self.add_output("shield R_out", units='m', lower=0)
        self.add_output("TF R_min",
                        units='m',
                        desc="TF leg casing minimum radius")
        self.add_output("TF R_in",
                        units='m',
                        desc="TF leg casing inner radius")

    def setup_partials(self):
        self.declare_partials("TF R_min", "Ob FW R_in", val=1)
        self.declare_partials("TF R_min", "blanket thickness", val=1)
        self.declare_partials("TF R_min", "access thickness", val=1)
        self.declare_partials("TF R_min", "VV thickness", val=1)
        self.declare_partials("TF R_min", "shield thickness", val=1)

        self.declare_partials("TF R_in", "Ob FW R_in", val=1)
        self.declare_partials("TF R_in", "blanket thickness", val=1)
        self.declare_partials("TF R_in", "access thickness", val=1)
        self.declare_partials("TF R_in", "VV thickness", val=1)
        self.declare_partials("TF R_in", "shield thickness", val=1)
        self.declare_partials("TF R_in", "gap thickness", val=1)

        self.declare_partials("blanket R_in", ["Ob FW R_in"], val=1)
        self.declare_partials("blanket R_out",
                              ["Ob FW R_in", "blanket thickness"],
                              val=1)
        self.declare_partials(
            "shield R_out",
            ["Ob FW R_in", "blanket thickness", "shield thickness"],
            val=1)

    def compute(self, inputs, outputs):
        ob_fw = inputs["Ob FW R_in"]
        ob_blanket_thickness = inputs["blanket thickness"]
        ob_shield_thickness = inputs["shield thickness"]
        ob_access_thickness = inputs["access thickness"]
        ob_vv_thickness = inputs["VV thickness"]

        outputs["blanket R_in"] = ob_fw
        blanket_r_out = ob_fw + ob_blanket_thickness
        outputs["blanket R_out"] = blanket_r_out
        outputs["shield R_out"] = blanket_r_out + ob_shield_thickness

        ob_build = [
            ob_fw, ob_blanket_thickness, ob_shield_thickness,
            ob_access_thickness, ob_vv_thickness
        ]
        outputs["TF R_min"] = sum(ob_build)
        outputs["TF R_in"] = outputs["TF R_min"] + inputs["gap thickness"]


class MenardSTOuterMachine(om.ExplicitComponent):
    r"""Radial build outside the TF coils

    A diagram of the radial build is shown below. On the left, (after the first
    entry) are the thicknesses of the various components. On the right are the
    absolute radii of various points (generally the inner or outer extent of
    components).::

                               Inputs   Outputs

                          Ob TF R_out +
                TF-cryostat thickness |
                                      + cryostat R_out

    Inputs
    ------
    Ob TF R_out : float
        m, Outboard TF leg outer radius
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
                       desc="Outboard TF leg outer radius")
        self.add_input("TF-cryostat thickness",
                       units="m",
                       desc="Outboard TF outer radius to cryostat outer wall")
        self.add_output("cryostat R_out",
                        units="m",
                        desc="Cryostat outer radius")

    def compute(self, inputs, outputs):
        TF_R_out = inputs["Ob TF R_out"]
        TF_to_cryostat_thickness = inputs["TF-cryostat thickness"]

        outputs["cryostat R_out"] = TF_R_out + TF_to_cryostat_thickness

    def setup_partials(self):
        self.declare_partials("cryostat R_out", "Ob TF R_out", val=1)
        self.declare_partials("cryostat R_out", "TF-cryostat thickness", val=1)


class STRadialBuild(om.Group):
    r"""The high-level radial build for the device.

    The device is built outward from the center. This group is designed so that
    all the inputs are positive widths, to ensure that the resulting device is
    always physically reasonable.

    The first component, CSToTF, builds the space for the CS out to the TF.
    Then, the InboardMagnetGeometry component specifies the build of the TF.
    The MenardSTInboard component finishes the inboard build out to the FW.
    The Plasma component determines the width to the outboard FW.
    The MenardSTOutboard component builds from the outboard FW to
      a minimum radius for the outboard TF leg.
    There is then an optional gap, in case the TF leg needs to be placed
    further out.
    The OutboardMagnetGeometry builds the outboard TF, and finally
    the MenardSTOuterMachine component builds the space out to the edge of the
    cryostat.

    Below are listed particularly interesting inputs and outputs.

    Inputs
    ------
    Plug R_out : float
        m, Central plug outer radius.
    CS ΔR : float
        m, CS thickness
    ib_tf.Δr_s : float
        m, Inboard TF inner structure thickness
    ib_tf.Δr_m : float
        m, Inboard TF winding pack thickness
    a : float
        m, Plasma minor radius

    Additional inputs to the 'ib', 'plasma', 'ob', and 'om' components
    are provided by the configuration file :code:`radial_build.yaml`.

    Outputs
    -------
    Ib TF R_in : float
        m, Inboard TF leg inner radius
    Ib TF R_out : float
        m, Inboard TF leg outer radius
    R0 : float
        m, Plasma major radius
    plasma R_out : float
        m, Outer radius of plasma at midplane
    A : float
        Plasma aspect ratio

    Ob TF R_min : float
        m, Outboard TF leg minimum inner radius
    cryostat R_out : float
        m, Outer radius of the cryostat
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        config = self.options['config']

        self.add_subsystem('props', Properties(config=config))
        self.add_subsystem('cs',
                           CSToTF(config=config),
                           promotes_inputs=["CS ΔR"],
                           promotes_outputs=[("TF R_in", "Ib TF R_in")])
        self.add_subsystem('ib_tf',
                           InboardMagnetGeometry(config=config),
                           promotes_inputs=[("r_is", "Ib TF R_in"), "n_coil"],
                           promotes_outputs=[("r_ot", "Ib TF R_out")])
        self.add_subsystem('ib',
                           MenardSTInboard(config=config),
                           promotes_inputs=[("TF R_out", "Ib TF R_out")])

        self.connect('props.Ib WC VV shield thickness',
                     ['ib.WC VV shield thickness'])
        self.connect('props.Ib FW thickness', ['ib.FW thickness'])

        self.add_subsystem('plasma',
                           Plasma(),
                           promotes_inputs=["a"],
                           promotes_outputs=[("R_in", "plasma R_in"),
                                             ("R_out", "plasma R_out"), "R0",
                                             "A", "Ob FW R_in"])
        self.connect('props.Ib SOL width', ['plasma.Ib SOL width'])
        self.connect('props.Ob SOL width', ['plasma.Ob SOL width'])
        self.connect('ib.FW R_out', ['plasma.Ib FW R_out'])

        self.add_subsystem('ob',
                           MenardSTOutboard(),
                           promotes_inputs=["Ob FW R_in"],
                           promotes_outputs=[("shield R_out",
                                              "Ob shield R_out"),
                                             ("TF R_min", "Ob TF R_min"),
                                             ("TF R_in", "Ob TF R_in")])
        self.connect('props.Ob shield thickness', ['ob.shield thickness'])
        self.connect('props.Ob access thickness', ['ob.access thickness'])
        self.connect('props.Ob vv thickness', ['ob.VV thickness'])

        self.add_subsystem('ob_tf',
                           OutboardMagnetGeometry(),
                           promotes_outputs=[("r_ov", "Ob TF R_out")])
        self.connect('ib_tf.Δr', 'ob_tf.Ib TF Δr')
        self.connect('Ob TF R_in', 'ob_tf.r_iu')

        # to be computed after TF thickness is determined
        self.add_subsystem('om',
                           MenardSTOuterMachine(),
                           promotes_inputs=["Ob TF R_out"],
                           promotes_outputs=["cryostat R_out"])
        self.connect('props.TF-cryostat thickness',
                     ['om.TF-cryostat thickness'])

        # constructs distances from the plasma to various outboard components
        # especially in order to use for use with 'parallel' curves.
        self.add_subsystem("ob_plasma_to_x",
                           om.ExecComp([
                               "ob_pl_to_fw_rin = ob_fw_rin - ob_pl_rout",
                               "ob_pl_to_bl_rout = ob_bl_rout - ob_pl_rout",
                               "ob_pl_to_sh_rout = ob_sh_rout - ob_pl_rout"
                           ],
                                       ob_pl_to_fw_rin={'units': 'm'},
                                       ob_pl_to_bl_rout={'units': 'm'},
                                       ob_pl_to_sh_rout={'units': 'm'},
                                       ob_fw_rin={'units': 'm'},
                                       ob_bl_rout={'units': 'm'},
                                       ob_sh_rout={'units': 'm'},
                                       ob_pl_rout={'units': 'm'}),
                           promotes_inputs=[("ob_pl_rout", "plasma R_out"),
                                            ("ob_fw_rin", "Ob FW R_in"),
                                            ("ob_sh_rout", "Ob shield R_out")],
                           promotes_outputs=[])
        self.connect("ob.blanket R_out", "ob_plasma_to_x.ob_bl_rout")


class MenardSTRadialBuild(om.Group):
    r"""The high-level radial build for the device.

    In this model, the inboard blanket and shield thicknesses are set as a
    function of aspect ratio.
    This is done to align with Menard's spreadsheet model.

    A nonlinear solver needs to be attached to this component.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        config = self.options['config']

        self.add_subsystem('radial_build',
                           STRadialBuild(config=config),
                           promotes_inputs=["a"],
                           promotes_outputs=["*"])

        self.add_subsystem("ib_blanket",
                           MenardInboardBlanketFit(config=config),
                           promotes_inputs=["A"])
        self.add_subsystem("ib_shield",
                           MenardInboardShieldFit(config=config),
                           promotes_inputs=["A"])
        self.add_subsystem("ob_blanket",
                           OutboardBlanketFit(config=config),
                           promotes_inputs=["A"])

        self.connect('ib_shield.shield_thickness',
                     ['radial_build.ib.WC shield thickness'])
        self.connect('ib_blanket.blanket_thickness',
                     ['radial_build.ib.blanket thickness'])
        self.connect('ob_blanket.blanket_thickness',
                     ['radial_build.ob.blanket thickness'])


if __name__ == "__main__":

    # test building a tokamak with A=1.6
    class MyRadialBuild(om.Group):
        def initialize(self):
            self.options.declare('config', default=None, recordable=False)

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
    prob.set_val("mrb.radial_build.CS ΔR", 0.105)
    prob.set_val("mrb.radial_build.ib_tf.Δr_s", 0.120)
    prob.set_val("mrb.radial_build.ib_tf.Δr_m", 0.100)
    prob.set_val("mrb.radial_build.ob.gap thickness", 0.100)

    model = prob.model
    newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    model.linear_solver = om.DirectSolver()

    prob.run_driver()
    all_inputs = prob.model.list_inputs(val=True, units=True, desc=True)
    all_outputs = prob.model.list_outputs(val=True, units=True, desc=True)
