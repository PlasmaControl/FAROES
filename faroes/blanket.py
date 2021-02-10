import numpy as np
from openmdao.utils.assert_utils import assert_check_partials
import openmdao.api as om
from scipy.constants import pi
from faroes.configurator import UserConfigurator, Accessor


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


class InboardMidplaneNeutronFluxFromRing(om.ExplicitComponent):
    r"""
    Assumes that all neutrons are sourced from a ring at the height of the
    midplane..
    Assumes an isotropic neutron source.
    Does not account for transparency or scattering (in solids or otherwise).

    Inputs
    ------
    S : float
        s**-1, Neutrons per second from the ring source
    P_n : float
        MW, Total neutron power
    R0 : float
        m, Radius of neutron source ring. Not necessarily
           the same as the plasma major radius.
    r_in : float
        m, radius

    Outputs
    -------
    Γ : float
        m**-2 s**-1, Neutron number flux at inboard midplane
    q_n : float
        MW / m**2, Neutron energy flux at inboard midplane
    """
    def setup(self):
        self.add_input("S", units="s**-1", val=0)
        self.add_input("P_n", units="MW", val=0)
        self.add_input("R0", units="m")
        self.add_input("r_in", units="m")
        self.add_output("Γ", units="m**-2 * s**-1")
        self.add_output("q_n", units="MW * m**-2")
        # coefficients for taylor expansion of
        # the 2 Ai * shape factor
        self.c = [3.126, 1.6868, 0.07742]
        self.Ai_0 = 2

    def compute(self, inputs, outputs):
        S = inputs["S"]
        P_n = inputs["P_n"]
        R0 = inputs["R0"]
        r_in = inputs["r_in"]
        Ai = R0 / (R0 - r_in)  # inboard "aspect ratio"
        ai = R0 - r_in  # inboard "minor radius"
        # for an infinite straight circular tube geometry
        SA = 2 * pi * ai * 2 * pi * R0
        Γ_tube = S / SA
        q_tube = P_n / SA
        c = self.c
        Ai_0 = self.Ai_0
        shape_factor = (c[0] + c[1] * (Ai - Ai_0) + c[2] *
                        (Ai - Ai_0)**2) / (2 * Ai)
        Γ = Γ_tube * shape_factor
        q_n = q_tube * shape_factor
        outputs["Γ"] = Γ
        outputs["q_n"] = q_n

    def setup_partials(self):
        self.declare_partials("Γ", ["S", "R0", "r_in"])
        self.declare_partials("q_n", ["P_n", "R0", "r_in"])

    def compute_partials(self, inputs, J):
        S = inputs["S"]
        P_n = inputs["P_n"]
        R0 = inputs["R0"]
        r_in = inputs["r_in"]
        Ai = R0 / (R0 - r_in)  # inboard "aspect ratio"
        ai = R0 - r_in  # inboard "minor radius"
        # for an infinite straight circular tube geometry
        SA = 2 * pi * ai * 2 * pi * R0
        Γ_tube = S / SA
        q_tube = P_n / SA
        c = self.c
        Ai_0 = self.Ai_0
        shape_factor = (c[0] + c[1] * (Ai - Ai_0) + c[2] *
                        (Ai - Ai_0)**2) / (2 * Ai)

        dshape_dAi = (-c[0] + Ai**2 * c[2] + Ai_0 *
                      (c[1] - Ai_0 * c[2])) / (2 * Ai**2)
        dAi_dR = -r_in / (R0 - r_in)**2
        dAi_dr_in = R0 / (R0 - r_in)**2

        dΓtube_dR0 = S * (r_in - 2 * R0) / (4 * pi**2 * R0**2 * (R0 - r_in)**2)
        dΓtube_dr_in = S / (4 * pi**2 * R0 * (R0 - r_in)**2)

        dqtube_dR0 = P_n * (r_in - 2 * R0) / (4 * pi**2 * R0**2 *
                                              (R0 - r_in)**2)
        dqtube_dr_in = P_n / (4 * pi**2 * R0 * (R0 - r_in)**2)

        J["Γ", "S"] = shape_factor / SA
        J["Γ", "R0"] = dΓtube_dR0 * shape_factor + Γ_tube * dshape_dAi * dAi_dR
        J["Γ", "r_in"] = dΓtube_dr_in * shape_factor \
            + Γ_tube * dshape_dAi * dAi_dr_in

        J["q_n", "P_n"] = shape_factor / SA
        J["q_n",
          "R0"] = dqtube_dR0 * shape_factor + q_tube * dshape_dAi * dAi_dR
        J["q_n", "r_in"] = dqtube_dr_in * shape_factor \
            + q_tube * dshape_dAi * dAi_dr_in


class MenardMagnetCoolingProperties(om.ExplicitComponent):
    r"""Loads default magnet cryogenics properties

    Outputs
    -------
    T_cold : float
        K, Cold temperature
    T_hot : float
        K, Exhaust temperature
    FOM : float
        Figure of merit; amount that it is worse than Carnot
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        acc = Accessor(self.options["config"])
        f = acc.accessor(["machine", "magnet cryogenics"])
        acc.set_output(self, f, "T_hot", units="K")
        acc.set_output(self, f, "T_cold", units="K")
        acc.set_output(self, f, "FOM")


class RefrigerationPerformance(om.ExplicitComponent):
    r"""Non-ideal refrigerator performance

    Inputs
    ------
    T_cold : float
        K, Cold temperature
    T_hot : float
        K, Exhaust temperature
    FOM : float
        Figure of merit; amount that it is worse than Carnot

    Outputs
    -------
    f_carnot : float
       Ideal factor. Inverse of the ideal Coefficent of Performance (COP)
    f : float
       Inverse of the Coefficient of Performance
    """
    def setup(self):
        self.add_input("T_cold", units="K")
        self.add_input("T_hot", units="K", val=300)
        self.add_input("FOM", val=1)
        self.add_output("f_carnot", lower=0, ref=10)
        self.add_output("f", lower=0, ref=20)

    def compute(self, inputs, outputs):
        T_c = inputs["T_cold"]
        T_h = inputs["T_hot"]
        FOM = inputs["FOM"]
        f_carnot = (T_h - T_c) / T_c
        f = f_carnot / FOM
        outputs["f_carnot"] = f_carnot
        outputs["f"] = f

    def setup_partials(self):
        self.declare_partials("f_carnot", ["T_cold", "T_hot"])
        self.declare_partials("f", ["T_cold", "T_hot", "FOM"])

    def compute_partials(self, inputs, J):
        T_c = inputs["T_cold"]
        T_h = inputs["T_hot"]
        FOM = inputs["FOM"]
        f_carnot = (T_h - T_c) / T_c
        J["f_carnot", "T_hot"] = 1 / T_c
        J["f_carnot", "T_cold"] = -T_h / T_c**2
        J["f", "T_hot"] = 1 / T_c / FOM
        J["f", "T_cold"] = -T_h / T_c**2 / FOM
        J["f", "FOM"] = -f_carnot / FOM**2


class MenardMagnetCooling(om.ExplicitComponent):
    r"""Required magnet cooling powers

    .. math::
        P_\mathrm{absorbed} = P_n * 0.06 * \exp(-{\Delta}r_{sh} / 0.081)

    Inputs
    ------
    Δr_sh : float
        m, Effective neutron shield thickness
    P_n : float
        MW, neutron power
    f_refrigeration : float
        Electric energy required to remove a unit of thermal energy from the
        cryogenic magnets. (This value is greater than 1.)

    Outputs
    -------
    shielding : float
        Factor by which the magnets are shielded from the neutron energy flux
    P_h : float
        MW, Neutron power that heats the magnets
    P_c,el : float
        MW, Electric power to deliver cooling at cryogenic temperatures
    """
    def setup(self, ):
        self.add_input("Δr_sh", units="m", val=0.6)
        self.add_input("P_n", units="MW")
        self.add_input("f_refrigeration", val=1)
        sh_ref = 1e-4
        self.add_output("shielding", val=sh_ref, ref=sh_ref, lower=0, upper=1)
        self.add_output("P_h", units="MW")
        self.add_output("P_c,el", units="MW")
        self.λ = 0.081  # decay length, m
        self.c = 0.06  # base power factor

    def compute(self, inputs, outputs):
        Δr = inputs["Δr_sh"]
        P_n = inputs["P_n"]
        f_cryo = inputs["f_refrigeration"]
        λ = self.λ
        c = self.c
        shielding = c * np.exp(-Δr / λ)
        P_h = P_n * shielding
        P_c = P_h * f_cryo
        outputs["shielding"] = shielding
        outputs["P_h"] = P_h
        outputs["P_c,el"] = P_c

    def setup_partials(self):
        self.declare_partials("shielding", ["Δr_sh"])
        self.declare_partials("P_h", ["Δr_sh", "P_n"])
        self.declare_partials("P_c,el", ["Δr_sh", "P_n", "f_refrigeration"])

    def compute_partials(self, inputs, J):
        Δr = inputs["Δr_sh"]
        P_n = inputs["P_n"]
        f_cryo = inputs["f_refrigeration"]
        λ = self.λ
        c = self.c
        shielding = c * np.exp(-Δr / λ)
        P_h = P_n * shielding
        J["shielding", "Δr_sh"] = -(c / λ) * np.exp(-Δr / λ)
        J["P_h", "Δr_sh"] = P_n * J["shielding", "Δr_sh"]
        J["P_h", "P_n"] = c * np.exp(-Δr / λ)
        J["P_c,el", "Δr_sh"] = -f_cryo * (c / λ) * P_n * np.exp(-Δr / λ)
        J["P_c,el", "P_n"] = f_cryo * c * np.exp(-Δr / λ)
        J["P_c,el", "f_refrigeration"] = P_h


class MagnetCryoCoolingPower(om.Group):
    r"""
    Inputs
    ------
    Δr_sh : float
        m, Effective neutron shield thickness
    P_n : float
        MW, neutron power

    Outputs
    -------
    T_cold : float
        K, Cold temperature
    f : float
       Inverse of the Coefficient of Performance
    shielding : float
        Factor by which the magnets are shielded from the neutron energy flux
    P_h : float
        MW, Neutron power that heats the magnets
    P_c,el : float
        MW, Electric power to deliver cooling at cryogenic temperatures
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           MenardMagnetCoolingProperties(config=config),
                           promotes_outputs=["T_cold"])
        self.add_subsystem("refrig_eff",
                           RefrigerationPerformance(),
                           promotes_inputs=["T_cold"])
        self.connect("props.T_hot", "refrig_eff.T_hot")
        self.connect("props.FOM", "refrig_eff.FOM")
        self.add_subsystem("power",
                           MenardMagnetCooling(),
                           promotes_inputs=["Δr_sh", "P_n"],
                           promotes_outputs=["P_c,el"])
        self.connect("refrig_eff.f", "power.f_refrigeration")


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()

    prob.model = MagnetCryoCoolingPower(config=uc)
    prob.setup(force_alloc_complex=True)

    prob.set_val("Δr_sh", 0.6)
    prob.set_val("P_n", 278.2)

    check = prob.check_partials(out_stream=None, method='cs')
    assert_check_partials(check)
    prob.run_driver()
    all_outputs = prob.model.list_outputs(values=True)

    # prob = om.Problem()

    # prob.model = MenardSTBlanketAndShieldGeometry()
    # prob.setup(force_alloc_complex=True)

    # prob.set_val('a', 1.1)
    # prob.set_val('κ', 2.7)
    # prob.set_val('Ob blanket R_out', 4.8)
    # prob.set_val('Ob blanket R_in', 4.0)
    # prob.set_val('Ib blanket R_out', 1.0)
    # prob.set_val('Ib blanket R_in', 0.8)
    # prob.set_val('Ob SOL width', 0.15)

    # prob.set_val('Ib WC shield R_out', 0.95)
    # prob.set_val('Ib WC shield R_in', 0.55)

    # prob.set_val('Ib WC VV shield R_out', 0.45)
    # prob.set_val('Ib WC VV shield R_in', 0.35)

    # check = prob.check_partials(out_stream=None, method='cs')
    # assert_check_partials(check)
    # prob.run_driver()
    # all_outputs = prob.model.list_outputs(values=True)
