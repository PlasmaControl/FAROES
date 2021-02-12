import faroes.util as util
from faroes.configurator import Accessor

import openmdao.api as om
from openmdao.utils.cs_safe import arctan2 as cs_safe_arctan2
from scipy.constants import pi
import numpy as np
from numpy import sin, cos, tan


class ThreeArcDeeTFSet(om.ExplicitComponent):
    r"""Three-arc Dee magnet set

    This magnet has a profile composed of a vertical line
    (the inner leg), two quarter-circle arcs, and an outer
    half-ellipse arc.

    Inputs
    ------
    R0 : float
        m, plasma major radius
    Ib TF R_out : float
        m, inboard leg outer radius
    hhs : float
        m, half-height of the straight segment
    e_a : float
        m, elliptical arc horizontal semi-major axis
    r_c : float
        m, circular arc radius
    θ   : array
        Angles relative to the magnetic axis at which to
           evaluate the distance to the magnet.

    Outputs
    -------
    e_b : float
        m, elliptical arc vertical semi-major axis
    d_sq : float
        m**2, Squared distance to points on the inner perimeter,
           at angle θ from the magnetic axis
    arc length: float
        m, inner perimeter of the magnet
    V_enc : float
        m**3, magnetized volume enclosed by the set
    """
    def setup(self):
        self.add_input("R0", units="m", desc="Plasma major radius")
        self.add_input("Ib TF R_out",
                       units="m",
                       desc="Inboard TF leg outer radius")
        self.add_input("e_a",
                       units="m",
                       desc="Elliptical arc horizontal semi-major axis")
        self.add_input("hhs",
                       units="m",
                       desc="Half-height of the straight segment")
        self.add_input("r_c", units="m", desc="Circular arc radius")
        self.add_input("θ", shape_by_conn=True)

        self.add_output("d_sq", units="m**2", copy_shape="θ")
        self.add_output("e_b",
                        units="m",
                        desc="Elliptical arc vertical semi-major axis")
        self.add_output("arc length",
                        units="m",
                        desc="Inner perimeter of the magnet")
        V_enc_ref = 1e3
        self.add_output("V_enc",
                        units="m**3",
                        lower=0,
                        ref=V_enc_ref,
                        desc="magnetized volume enclosed by the set")

    def compute(self, inputs, outputs):
        R0 = inputs["R0"]
        r_ot = inputs["Ib TF R_out"]

        e_a = inputs["e_a"]
        hhs = inputs["hhs"]
        r_c = inputs["r_c"]
        θ_all = inputs["θ"]
        e_b = hhs + r_c
        outputs["e_b"] = e_b

        arc_length = (util.ellipse_perimeter_ramanujan(e_a, e_b) / 2 +
                      pi * r_c + 2 * hhs)
        outputs["arc length"] = arc_length

        # radius of the transition between the quarter-circles and ellipse
        R = r_ot + r_c
        v_1 = util.half_ellipse_torus_volume(R, e_a, e_b)
        v_2 = util.half_ellipse_torus_volume(R, -r_c, r_c)
        v_3 = pi * 2 * hhs * (R**2 - (R - r_c)**2)
        outputs["V_enc"] = v_1 + v_2 + v_3

        θ1 = cs_safe_arctan2(e_b, R - R0)
        θ2 = cs_safe_arctan2(hhs, r_ot - R0)
        θ3 = cs_safe_arctan2(-hhs, r_ot - R0)
        θ4 = cs_safe_arctan2(-e_b, R - R0)

        θ = θ_all
        on_ellipse = (θ4 < θ) * (θ < θ1)
        on_upper_circ = (θ1 <= θ) * (θ < θ2)
        on_lower_circ = (θ3 < θ) * (θ <= θ4)
        on_upper_straight = θ2 <= θ
        on_lower_straight = θ <= θ3
        # abbreviations for easier writing

        θ = θ_all[on_lower_straight]
        d2_lower_straight = (R0 - r_ot)**2 / cos(θ)**2

        θ = θ_all[on_lower_circ]

        a = e_a
        b = e_b
        c = R - R0  # x-distance between ellipse center and plasma center
        lower_circ_root = np.sqrt(-2 * (c**2 + hhs**2 - 2 * r_c**2) + 2 *
                                  (c**2 + hhs**2) *
                                  cos(2 * θ - 2 * cs_safe_arctan2(-hhs, c)))
        d_lower_circ = c * cos(θ) + hhs * sin(θ) - (1 / 2) * lower_circ_root
        d2_lower_circ = d_lower_circ**2


        θ = θ_all[on_ellipse]
        t = tan(θ)
        n1 = b**2 * (b * c + a * (b**2 + (a**2 - c**2) * t**2)**(1 / 2))**2
        n2 = b**2 * ((b**2 + a**2 * t**2)**2 -
                     (a * c * t**2 - b * (b**2 +
                                          (a**2 - c**2) * t**2)**(1 / 2))**2)
        numerator = n1 + n2
        denominator = (b**2 + a**2 * t**2)**2
        d2_ell = numerator / denominator

        θ = θ_all[on_upper_circ]
        upper_circ_root = np.sqrt(-2 * (c**2 + hhs**2 - 2 * r_c**2) + 2 *
                                  (c**2 + hhs**2) *
                                  cos(2 * θ - 2 * cs_safe_arctan2(hhs, c)))
        d_upper_circ = c * cos(θ) + hhs * sin(θ) + (1 / 2) * upper_circ_root
        d2_upper_circ = d_upper_circ**2

        θ = θ_all[on_upper_straight]
        d2_upper_straight = (R0 - r_ot)**2 / cos(θ)**2

        d2 = np.hstack([
            d2_lower_straight, d2_lower_circ, d2_ell, d2_upper_circ,
            d2_upper_straight
        ])

        outputs["d_sq"] = d2

    def setup_partials(self):
        size = self._get_var_meta("θ", "size")
        self.declare_partials("e_b", ["hhs", "r_c"], val=1)
        self.declare_partials("arc length", ["e_a", "hhs", "r_c"])
        self.declare_partials("V_enc", ["Ib TF R_out", "e_a", "hhs", "r_c"])
        self.declare_partials("d_sq", ["Ib TF R_out", "e_a", "hhs", "r_c"], method="cs")
        self.declare_partials("d_sq", ["R0"], method="cs")
        #self.declare_partials("d_sq", ["θ"], rows=range(size), cols=range(size))
        self.declare_partials("d_sq", ["θ"], method="cs")

    def compute_partials(self, inputs, J):
        size = self._get_var_meta("θ", "size")
        R0 = inputs["R0"]
        r_ot = inputs["Ib TF R_out"]

        e_a = inputs["e_a"]
        hhs = inputs["hhs"]
        r_c = inputs["r_c"]
        e_b = hhs + r_c

        deb_dhhs = 1
        deb_drc = 1

        dfullarclen = util.ellipse_perimeter_ramanujan_derivatives(e_a, e_b)
        darclen_dea = dfullarclen["a"] / 2
        darclen_deb = dfullarclen["b"] / 2
        J["arc length", "e_a"] = darclen_dea
        J["arc length", "hhs"] = darclen_deb * deb_dhhs + 2
        J["arc length", "r_c"] = darclen_deb * deb_drc + pi

        R = r_ot + r_c
        dR_drot = 1
        dR_drc = 1
        dv1 = util.half_ellipse_torus_volume_derivatives(R, e_a, e_b)
        dv1_dR, dv1_dea, dv1_deb = dv1["R"], dv1["a"], dv1["b"]
        dv2 = util.half_ellipse_torus_volume_derivatives(R, -r_c, r_c)
        dv2_dR, dv2_drc1, dv2_drc2 = dv2["R"], dv2["a"], dv2["b"]

        dv3_dhhs = 2 * pi * (R**2 - (R - r_c)**2)
        dv3_dR = 4 * pi * hhs * r_c
        dv3_drc = 4 * pi * hhs * (R - r_c)

        J["V_enc", "e_a"] = dv1_dea
        J["V_enc", "hhs"] = dv1_deb * deb_dhhs + dv3_dhhs
        J["V_enc",
          "r_c"] = (dv1_deb * deb_drc + dv1_dR * dR_drc - dv2_drc1 + dv2_drc2 +
                    dv2_dR * dR_drc + dv3_drc + dv3_dR * dR_drc)
        J["V_enc", "Ib TF R_out"] = (dv1_dR * dR_drot + dv2_dR * dR_drot +
                                     dv3_dR * dR_drot)

        #θ_all = inputs["θ"]
        #θ1 = cs_safe_arctan2(e_b, R - R0)
        #θ2 = cs_safe_arctan2(hhs, r_ot - R0)
        #θ3 = cs_safe_arctan2(-hhs, r_ot - R0)
        #θ4 = cs_safe_arctan2(-e_b, R - R0)

        ## abbreviation for easier writing
        #θ = θ_all
        #on_ellipse = (θ4 < θ) * (θ < θ1)
        #on_upper_circ = (θ1 <= θ) * (θ < θ2)
        #on_lower_circ = (θ3 < θ) * (θ <= θ4)
        #on_upper_straight = θ2 <= θ
        #on_lower_straight = θ <= θ3

        #θ = θ_all[on_lower_straight]
        #d2_lower_straight = (R0 - r_ot)**2 / cos(θ)**2

        #dd2_dR0 = np.zeros(size)
        #dd2_dR0[on_lower_straight] = 2 * (R0 - r_ot) / cos(θ)**2
        #J["d_sq", "R0"] = dd2_dR0
        #print(dd2_dR0)




if __name__ == "__main__":
    prob = om.Problem()

    θ = np.linspace(-pi, pi, 31, endpoint=True)

    prob.model.add_subsystem("ivc",
                             om.IndepVarComp("θ", val=θ),
                             promotes_outputs=["*"])

    prob.model.add_subsystem("tadTF",
                             ThreeArcDeeTFSet(),
                             promotes_inputs=["*"])

    prob.setup()

    prob.set_val("Ib TF R_out", 1.94)
    prob.set_val("R0", 4)
    prob.set_val("r_c", 1.6)
    prob.set_val("e_a", 3.88)
    prob.set_val("hhs", 2.72)
    # prob.set_val("θ", [1,2,3])
    # prob.set_val("cross section", 0.052)
    # prob.set_val("n_coil", 18)

    prob.run_driver()
    #all_inputs = prob.model.list_inputs(print_arrays=True)
    all_outputs = prob.model.list_outputs(print_arrays=True)
