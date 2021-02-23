import faroes.util as util

import openmdao.api as om
from openmdao.utils.cs_safe import arctan2 as cs_safe_arctan2
from scipy.constants import pi
import numpy as np


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
    half-height : float
        m, half the vertical height of the conductors
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

        self.add_output("e_b",
                        units="m",
                        desc="Elliptical arc vertical semi-major axis")
        self.add_output("arc length",
                        units="m",
                        desc="Inner perimeter of the magnet")
        V_enc_ref = 1e3
        self.add_output("Ob TF R_in", units="m", ref=10, lower=0)
        self.add_output("constraint_axis_within_coils", units="m", ref=1)
        self.add_output("V_enc",
                        units="m**3",
                        lower=0,
                        ref=V_enc_ref,
                        desc="magnetized volume enclosed by the set")
        self.add_output("d_sq", units="m**2", copy_shape="θ")
        self.add_output("half-height",
                        units="m", lower=0,
                        desc="Average semi-major axis of the magnet")

    def compute(self, inputs, outputs):
        size = self._get_var_meta("θ", "size")
        R0 = inputs["R0"]
        r_ot = inputs["Ib TF R_out"]

        e_a = inputs["e_a"]
        hhs = inputs["hhs"]
        r_c = inputs["r_c"]
        θ_all = inputs["θ"]
        e_b = hhs + r_c
        outputs["e_b"] = e_b
        outputs["Ob TF R_in"] = r_ot + r_c + e_a
        outputs["constraint_axis_within_coils"] = r_ot + r_c + e_a - R0

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
        on_straight = np.logical_or(θ2 <= θ, θ <= θ3)
        # abbreviations for easier writing

        d2 = np.zeros(size, dtype=np.cdouble)

        θ = θ_all[on_straight]
        d2[on_straight] = (R0 - r_ot)**2 / np.cos(θ)**2

        θ = θ_all[on_lower_circ]
        d = util.polar_offset_ellipse(a=r_c, b=r_c, x=R - R0, y=-hhs, t=θ)
        d2[on_lower_circ] = d**2

        θ = θ_all[on_upper_circ]
        d = util.polar_offset_ellipse(a=r_c, b=r_c, x=R - R0, y=+hhs, t=θ)
        d2[on_upper_circ] = d**2

        # distance to ellipse
        θ = θ_all[on_ellipse]
        d = util.polar_offset_ellipse(a=e_a, b=e_b, x=R - R0, y=0, t=θ)
        d2[on_ellipse] = d**2

        outputs["d_sq"] = d2

        outputs["half-height"] = e_b

    def setup_partials(self):
        self.declare_partials("e_b", ["hhs", "r_c"], val=1)
        self.declare_partials("half-height", ["hhs", "r_c"], val=1)
        self.declare_partials("arc length", ["e_a", "hhs", "r_c"])
        self.declare_partials("V_enc", ["Ib TF R_out", "e_a", "hhs", "r_c"])
        self.declare_partials("d_sq", ["Ib TF R_out", "e_a", "hhs", "r_c"],
                              method="cs")
        self.declare_partials("d_sq", ["R0"], method="cs")
        self.declare_partials("d_sq", ["θ"], method="cs")
        self.declare_partials("Ob TF R_in", ["e_a", "r_c", "Ib TF R_out"],
                              val=1)
        self.declare_partials("constraint_axis_within_coils",
                              ["e_a", "r_c", "Ib TF R_out"],
                              val=1)
        self.declare_partials("constraint_axis_within_coils", ["R0"], val=-1)

    def compute_partials(self, inputs, J):
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

    def plot(self, ax=None, **kwargs):
        size = 100
        color = "black"

        if "color" in kwargs.keys():
            color = kwargs[color]

        t = np.linspace(0, 1, 100)
        r_ot = self.get_val("Ib TF R_out")
        straight_R = r_ot * np.ones(size)
        hhs = self.get_val("hhs")
        straight_Z = t * (2 * hhs) - hhs
        ax.plot(straight_R, straight_Z, color=color, **kwargs)

        # quarter circles
        r_c = self.get_val("r_c")
        c_R = r_ot + r_c * (1 - np.cos(t * np.pi / 2))
        c_Z = hhs + r_c * (np.sin(t * np.pi / 2))
        ax.plot(c_R, c_Z, color=color, **kwargs)
        ax.plot(c_R, -c_Z, color=color, **kwargs)

        # half-ellipse
        e_a = self.get_val("e_a")
        e_b = self.get_val("e_b")
        el_R = r_ot + r_c + e_a * np.cos(t * np.pi - np.pi / 2)
        el_Z = e_b * np.sin(t * np.pi - np.pi / 2)
        ax.plot(el_R, el_Z, color=color, **kwargs)


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
    # all_inputs = prob.model.list_inputs(print_arrays=True)
    all_outputs = prob.model.list_outputs(print_arrays=True)
