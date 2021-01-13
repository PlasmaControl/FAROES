import numpy as np
from openmdao.utils.assert_utils import assert_check_partials
import openmdao.api as om
from scipy.constants import pi


class MenardSTBlanketAndShieldMagnetProtection(om.ExplicitComponent):
    r"""Shielding properties of the blanket and shield for the Ib magnets

    Inputs
    ------
    Ib blanket thickness : float
        m, Inboard blanket thickness
    Ib WC shield thickness : float
        m, Inboard WC shield thickness
    Ib WC VV shield thickness : float
        m, Inboard inter-VV WC shield thickness

    Outputs
    -------
    Eff Sh+Bl n thickness : float
        m, Effective neutron-stopping length of inboard blanket and shield
    Total WC Sh thickness : float
        m, Total thickness of the inboard blanket and shield
    Total Sh+Bl thickness : float
        m, Total thickness of the inboard blanket and shield
    Shielding factor : float
        Factor by which shielding is better than the reference case
    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            config = self.options["config"]
            f = config.accessor(["materials", "blanket"])
            self.rel_shield = f(['relative shielding'])
            f = config.accessor(["materials", "HFS neutron shield"])
            self.ccfe_ref_thick = f(['CCFE reference thickness'], units='m')
            self.tenx_decay_len = f(['10x decay length'], units='m')
        else:
            self.rel_shield = 0.5
            self.ccfe_ref_thick = 0.57
            self.tenx_decay_len = 0.155

        self.add_input('Ib blanket thickness',
                       units='m',
                       desc="Inboard blanket thickness")
        self.add_input('Ib WC shield thickness',
                       units='m',
                       desc="Inboard WC shield thickness")
        self.add_input('Ib WC VV shield thickness',
                       units='m',
                       desc="Inboard inter-VV WC shield thickness")

        self.add_output(
            'Eff Sh+Bl n thickness',
            units='m',
            lower=0,
            desc="Effective neutron-stopping length of Ib blanket and shield")
        self.add_output(
            "Total Sh+Bl thickness",
            units="m",
            lower=0,
            desc="Total thickness of the inboard blanket and shield")
        self.add_output("Total WC Sh thickness",
                        units="m",
                        desc="Total thickness of the inboard WC shield")
        self.add_output(
            "Shielding factor",
            lower=0,
            desc="Factor by which shielding is better than the reference case")

    def compute(self, inputs, outputs):
        bl_thick = inputs["Ib blanket thickness"]
        sh_thick = inputs["Ib WC shield thickness"]
        vv_sh_thick = inputs["Ib WC VV shield thickness"]
        outputs["Total Sh+Bl thickness"] = bl_thick + sh_thick + vv_sh_thick
        outputs["Total WC Sh thickness"] = sh_thick + vv_sh_thick
        eff_bl_thick = self.rel_shield * bl_thick + outputs[
            "Total WC Sh thickness"]
        outputs["Eff Sh+Bl n thickness"] = eff_bl_thick

        shf = 10**((eff_bl_thick - self.ccfe_ref_thick) / self.tenx_decay_len)
        outputs["Shielding factor"] = shf

    def setup_partials(self):
        self.declare_partials("Total Sh+Bl thickness",
                              "Ib blanket thickness",
                              val=1)
        self.declare_partials("Total Sh+Bl thickness",
                              "Ib WC shield thickness",
                              val=1)
        self.declare_partials("Total Sh+Bl thickness",
                              "Ib WC VV shield thickness",
                              val=1)
        self.declare_partials("Total WC Sh thickness",
                              "Ib WC shield thickness",
                              val=1)
        self.declare_partials("Total WC Sh thickness",
                              "Ib WC VV shield thickness",
                              val=1)
        self.declare_partials(
            "Eff Sh+Bl n thickness",
            ["Ib WC shield thickness", "Ib WC VV shield thickness"],
            val=1)
        self.declare_partials("Eff Sh+Bl n thickness",
                              ["Ib blanket thickness"])
        self.declare_partials("Shielding factor", [
            "Ib WC shield thickness", "Ib WC VV shield thickness",
            "Ib blanket thickness"
        ])

    def compute_partials(self, inputs, J):
        J["Eff Sh+Bl n thickness", "Ib blanket thickness"] = self.rel_shield

        bl_thick = inputs["Ib blanket thickness"]
        sh_thick = inputs["Ib WC shield thickness"]
        vv_sh_thick = inputs["Ib WC VV shield thickness"]
        eff_bl_thick = self.rel_shield * bl_thick + sh_thick + vv_sh_thick

        shf = 10**((eff_bl_thick - self.ccfe_ref_thick) / self.tenx_decay_len)
        dshf_deff = shf * np.log(10) / self.tenx_decay_len
        J["Shielding factor", "Ib WC shield thickness"] = dshf_deff * 1
        J["Shielding factor", "Ib WC VV shield thickness"] = dshf_deff * 1
        J["Shielding factor",
          "Ib blanket thickness"] = dshf_deff * self.rel_shield


class MenardSTBlanketAndShieldGeometry(om.ExplicitComponent):
    r"""
    Should be run after the inboard & outboard radial builds

    Inputs
    ------
    a : float
        m, plasma minor radius
    κ : float
        plasma elongation
    Ob SOL width : float
        m, Outboard SOL width
    Ib blanket R_out: float
        m, Inboard blanket outer radius
    Ib blanket R_in: float
        m, Inboard blanket inner radius

    Ob blanket R_in : float
        m, Outboard blanket inner radius
    Ob blanket R_out : float
        m, Outboard blanket outer radius

    Ib WC shield R_out : float
        m, WC shield outer radius
    Ib WC shield R_in : float
        m, WC shield inner radius

    Ib WC VV shield R_out : float
        m, Inter-VV tungsten carbide neutron shield outer radius
    Ib WC VV shield R_in : float
        m, Inter-VV tungsten carbide neutron shield inner radius

    Outputs
    -------
    Ib h : float
        m, Inboard blanket and shield height
    Ib blanket V : float
        m**3, Inboard blanket estimated volume
    Ob blanket V : float
        m**3, Outboard blanket estimated volume
    Ib shield V : float
        m**3, Inboard shield estimated volume
    Blanket V : float
        m**3, Total blanket estimated volume

    """
    def initialize(self):
        # there may be a bug in menard's calculation for shield volume
        # if so, change self.bug to 1
        self.bug = 0

    def setup(self):
        self.add_input("a", units="m", desc="Plasma minor radius")
        self.add_input("κ", desc="plasma elongation")
        self.add_input("Ob SOL width", units="m", desc="Outboard SOL width")
        self.add_input("Ib blanket R_out",
                       units="m",
                       desc="Inboard blanket outer radius")
        self.add_input("Ib blanket R_in",
                       units="m",
                       desc="Inboard blanket inner radius")
        self.add_input("Ob blanket R_out",
                       units="m",
                       desc="Outboard blanket outer radius")
        self.add_input("Ob blanket R_in",
                       units="m",
                       desc="Outboard blanket inner radius")

        self.add_input("Ib WC shield R_out",
                       units="m",
                       desc="WC shield outer radius")
        self.add_input("Ib WC shield R_in",
                       units="m",
                       desc="WC shield inner radius")
        self.add_input("Ib WC VV shield R_out",
                       units="m",
                       desc="Inter-VV WC shield outer radius")
        self.add_input("Ib WC VV shield R_in",
                       units="m",
                       desc="Inter-VV WC shield inner radius")

        V_ref = 100  # m³
        self.add_output("Ib h",
                        units="m",
                        desc="Inboard blanket and shield height")
        self.add_output("Ib blanket V",
                        units="m**3",
                        desc="Inboard blanket estimated volume")
        self.add_output("Ib shield V",
                        units="m**3",
                        desc="Inboard shield estimated volume")
        self.add_output("Ob blanket V",
                        units="m**3",
                        ref=V_ref,
                        desc="Outboard blanket estimated volume")
        self.add_output("Blanket V",
                        units="m**3",
                        ref=V_ref,
                        desc="Total blanket estimated volume")

    def compute(self, inputs, outputs):
        a = inputs["a"]
        κ = inputs["κ"]
        Ob_SOL_ΔR = inputs["Ob SOL width"]

        h = 2 * κ * (a + Ob_SOL_ΔR)
        outputs["Ib h"] = h

        R_out = inputs["Ib blanket R_out"]
        R_in = inputs["Ib blanket R_in"]
        outputs["Ib blanket V"] = pi * (R_out**2 - R_in**2) * h

        R_out = inputs["Ib WC VV shield R_out"]
        R_in = inputs["Ib WC VV shield R_in"]
        outputs["Ib shield V"] = pi * (R_out**2 - R_in**2) * h

        R_out = inputs["Ib WC shield R_out"]
        R_in = inputs["Ib WC shield R_in"]
        outputs["Ib shield V"] += self.bug * pi * (R_out**2 - R_in**2) * h

        # there seems to definitely be a bug in this so I'm going to do
        # something different
        R_out = inputs["Ob blanket R_out"]
        R_in = inputs["Ob blanket R_in"]
        outputs["Ob blanket V"] = pi * (R_out**2 - R_in**2) * h

        outputs[
            "Blanket V"] = outputs["Ib blanket V"] + outputs["Ob blanket V"]

    def setup_partials(self):
        self.declare_partials("Ib h", ["a", "κ", "Ob SOL width"])
        self.declare_partials(
            "Ib blanket V",
            ["a", "κ", "Ob SOL width", "Ib blanket R_out", "Ib blanket R_in"])
        self.declare_partials("Ib shield V", [
            "a", "κ", "Ob SOL width", "Ib WC shield R_in",
            "Ib WC shield R_out", "Ib WC VV shield R_out",
            "Ib WC VV shield R_in"
        ])
        self.declare_partials(
            "Ob blanket V",
            ["a", "κ", "Ob SOL width", "Ob blanket R_out", "Ob blanket R_in"])
        self.declare_partials("Blanket V", [
            "a", "κ", "Ob SOL width", "Ob blanket R_out", "Ob blanket R_in",
            "Ib blanket R_out", "Ib blanket R_in"
        ])

    def compute_partials(self, inputs, J):
        a = inputs["a"]
        κ = inputs["κ"]
        Ob_SOL_ΔR = inputs["Ob SOL width"]

        J["Ib h", "a"] = 2 * κ
        J["Ib h", "κ"] = 2 * (a + Ob_SOL_ΔR)
        J["Ib h", "Ob SOL width"] = 2 * κ

        R_out = inputs["Ib blanket R_out"]
        R_in = inputs["Ib blanket R_in"]

        # horizontal cross section
        h_CX = pi * (R_out**2 - R_in**2)

        J["Ib blanket V", "a"] = 2 * κ * h_CX
        J["Ib blanket V", "κ"] = 2 * h_CX * (a + Ob_SOL_ΔR)
        J["Ib blanket V", "Ob SOL width"] = 2 * κ * h_CX
        J["Ib blanket V",
          "Ib blanket R_out"] = 4 * pi * R_out * (a + Ob_SOL_ΔR) * κ
        J["Ib blanket V",
          "Ib blanket R_in"] = -4 * pi * R_in * (a + Ob_SOL_ΔR) * κ

        bug = self.bug
        R_m_out = inputs["Ib WC shield R_out"]
        R_m_in = inputs["Ib WC shield R_in"]
        R_vv_out = inputs["Ib WC VV shield R_out"]
        R_vv_in = inputs["Ib WC VV shield R_in"]

        # horizontal cross section
        h_m_CX = pi * (R_m_out**2 - R_m_in**2)
        h_vv_CX = pi * (R_vv_out**2 - R_vv_in**2)
        J["Ib shield V", "a"] = 2 * κ * (h_vv_CX + bug * h_m_CX)
        J["Ib shield V", "κ"] = 2 * (h_vv_CX + bug * h_m_CX) * (a + Ob_SOL_ΔR)
        J["Ib shield V", "Ob SOL width"] = J["Ib shield V", "a"]
        J["Ib shield V",
          "Ib WC shield R_out"] = 4 * bug * pi * R_m_out * κ * (a + Ob_SOL_ΔR)
        J["Ib shield V",
          "Ib WC shield R_in"] = -4 * bug * pi * R_m_in * κ * (a + Ob_SOL_ΔR)
        J["Ib shield V",
          "Ib WC VV shield R_out"] = 4 * pi * R_vv_out * κ * (a + Ob_SOL_ΔR)
        J["Ib shield V",
          "Ib WC VV shield R_in"] = -4 * pi * R_vv_in * κ * (a + Ob_SOL_ΔR)

        R_out = inputs["Ob blanket R_out"]
        R_in = inputs["Ob blanket R_in"]
        h_CX = pi * (R_out**2 - R_in**2)

        J["Ob blanket V", "a"] = 2 * κ * h_CX
        J["Ob blanket V", "κ"] = 2 * h_CX * (a + Ob_SOL_ΔR)
        J["Ob blanket V", "Ob SOL width"] = 2 * κ * h_CX
        J["Ob blanket V",
          "Ob blanket R_out"] = 4 * pi * R_out * (a + Ob_SOL_ΔR) * κ
        J["Ob blanket V",
          "Ob blanket R_in"] = -4 * pi * R_in * (a + Ob_SOL_ΔR) * κ

        J["Blanket V", "a"] = J["Ib blanket V", "a"] + J["Ob blanket V", "a"]
        J["Blanket V", "κ"] = J["Ib blanket V", "κ"] + J["Ob blanket V", "κ"]
        J["Blanket V", "Ob SOL width"] = J["Ib blanket V",
                                           "Ob SOL width"] + J["Ob blanket V",
                                                               "Ob SOL width"]
        J["Blanket V", "Ob blanket R_out"] = J["Ob blanket V",
                                               "Ob blanket R_out"]
        J["Blanket V", "Ob blanket R_in"] = J["Ob blanket V",
                                              "Ob blanket R_in"]
        J["Blanket V", "Ib blanket R_out"] = J["Ib blanket V",
                                               "Ib blanket R_out"]
        J["Blanket V", "Ib blanket R_in"] = J["Ib blanket V",
                                              "Ib blanket R_in"]


if __name__ == "__main__":
    prob = om.Problem()

    prob.model = MenardSTBlanketAndShieldGeometry()
    prob.setup(force_alloc_complex=True)

    prob.set_val('a', 1.1)
    prob.set_val('κ', 2.7)
    prob.set_val('Ob blanket R_out', 4.8)
    prob.set_val('Ob blanket R_in', 4.0)
    prob.set_val('Ib blanket R_out', 1.0)
    prob.set_val('Ib blanket R_in', 0.8)
    prob.set_val('Ob SOL width', 0.15)

    prob.set_val('Ib WC shield R_out', 0.95)
    prob.set_val('Ib WC shield R_in', 0.55)

    prob.set_val('Ib WC VV shield R_out', 0.45)
    prob.set_val('Ib WC VV shield R_in', 0.35)

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)
    prob.run_driver()
    all_outputs = prob.model.list_outputs(values=True)
