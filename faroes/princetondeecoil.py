import openmdao.api as om
from openmdao.utils.cs_safe import arctan2 as cs_safe_arctan2
from scipy.constants import pi
from scipy.integrate import solve_ivp
from scipy.special import iv, modstruve
import numpy as np


class PrincetonDeeTFSet(om.ExplicitComponent):
    r"""Simplified constant-tension coil shape

    This magnet has an inner profile composed of a vertical line
    (the inner leg), and a constant-tension shape.

    Notes
    -----
    Assumes a large number of filamentary currents. There are other solutions
    for finite numbers of coils but that is not implemented here.

    This is also not strictly correct because this solves for an inner
    perimeter of the constant-tension coil shape. Ideally this shape would be
    that of the conductors, which then have some casing around them.

    Inputs
    ------
    R0 : float
        m, plasma major radius
    Ib TF R_out : float
        m, inboard leg outer radius
    ΔR : float
        Magnet bore width. Must be greater than 0.
    θ   : array
        Angles relative to the magnetic axis at which to
           evaluate the distance to the magnet.

    Outputs
    -------
    k : float
        Normalized magnet shape parameter
    Ob TF R_in : float
        m, outboard leg inner radius.
    inner leg half-height : float
        m, half-height of the vertical inner leg
    d_sq : float
        m**2, Squared distance to points on the inner perimeter,
           at angle θ from the magnetic axis
    V_enc : float
        m**3, magnetized volume enclosed by the set
    arc length: float
        m, inner perimeter of the magnet
    half-height : float
        m, half the vertical height of the conductors

    constraint_axis_within_coils : float
        m, Constraint which is greater than 0 when the major axis
           is within the outer legs

    References
    ----------
    .. [1] Gralnick, S. L.; Tenney, F. H.
       Analytic Solutions for Constant‐tension Coil Shapes.
       J. Appl. Phys. 1976, 47, 7.
       https://doi.org/10.1063/1.322993
    """
    def setup(self):
        self.add_input("R0", units="m", desc="Plasma major radius")
        self.add_input("Ib TF R_out",
                       units="m",
                       desc="Inboard TF leg outer radius")
        self.add_input("ΔR", units="m", desc="Magnet bore width")
        self.add_input("θ", shape_by_conn=True)

        self.add_output("k", desc="Normalized magnet shape parameter")
        # self.add_output("arc length",
        #                 units="m",
        #                 desc="Inner perimeter of the magnet")
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
                        units="m",
                        lower=0,
                        desc="Average semi-major axis of the magnet")

    def inner_leg_half_height(self, k, r0):
        return r0 * k * pi * iv(1, k)

    def terminal_theta(self, k, r0, r1, x0):
        return cs_safe_arctan2(self.inner_leg_half_height(k, 1),
                               -(x0 - r1) / r0)

    def solve_curve(self, r1, r2, x0, θ_eval):
        r"""
        Parameters
        ----------
        r1 : float
           m, Inner leg radius
        r2 : float
           m, Outer leg radius
        x0 : float
           m, Plasma axis radius
        """
        # want only single floats here; I have not configured the ivp solver to
        # be vector-compatible
        r1 = r1[0]
        r2 = r2[0]
        x0 = x0[0]
        r0 = np.sqrt(r1 * r2)
        k = np.log(r2 / r1) / 2
        θ_end = self.terminal_theta(k, r0, r1, x0)

        def rhs(t, y, x0, k):
            y1, y2 = y
            numer = x0 * y1**2 + np.cos(
                t) * y1**3 + 2 * x0 * y2**2 + 2 * np.cos(t) * y1 * y2**2 - (
                    y1**2 + y2**2)**(3 / 2) / k
            denom = x0 * y1 + np.cos(t) * y1**2
            return [y2, numer / denom]

        sol = solve_ivp(fun=rhs,
                        args=(x0 / r0, k),
                        t_span=[0, θ_end],
                        y0=[r2 / r0 - x0 / r0, 0],
                        t_eval=θ_eval,
                        max_step=max((x0 / r0) / 1e1, 0.1))
        return {"ρ": r0 * sol["y"][0], "dρ/dθ": r0 * sol["y"][1]}

    def compute(self, inputs, outputs):
        size = self._get_var_meta("θ", "size")
        R0 = inputs["R0"]
        r_ot = inputs["Ib TF R_out"]

        # here r1 refers to the conductor radius,
        # while r_ot refers to the casing
        # radius. They are set to be equal.
        r1 = r_ot
        ΔR = inputs["ΔR"]

        # here r2 refers to the conductor radius,
        # while r_iu refers to the casing
        # radius. They are set to be equal.
        r_iu = r_ot + ΔR
        outputs["Ob TF R_in"] = r_iu
        r2 = r_iu
        r0 = np.sqrt(r1 * r2)

        k = np.log(r2 / r1) / 2
        outputs["k"] = k

        r0 = np.sqrt(r1 * r2)

        outputs["constraint_axis_within_coils"] = r_ot + ΔR - R0
        θ_all = inputs["θ"]

        # angle of the transition between the vertical inner leg and the curve
        θ_max_curve = self.terminal_theta(k, r0, r1, R0)

        θ = θ_all
        on_upper_curve = (θ >= 0) * (θ < θ_max_curve)
        on_lower_curve = (θ < 0) * (θ > -θ_max_curve)
        on_straight = np.logical_or(θ <= -θ_max_curve, θ >= θ_max_curve)

        d2 = np.zeros(size)

        θ = θ_all[on_straight]
        d2[on_straight] = (R0 - r_ot)**2 / np.cos(θ)**2

        θ = θ_all[on_upper_curve]
        sols = self.solve_curve(r1, r2, R0, θ)
        d2[on_upper_curve] = sols["ρ"]**2

        θ = θ_all[on_lower_curve]
        sols = self.solve_curve(r1, r2, R0, -θ[::-1])
        d2[on_lower_curve] = sols["ρ"][::-1]**2

        outputs["d_sq"] = d2

        # compute v_enc
        outputs["V_enc"] = 2 * r0**3 * k * pi**2 * (iv(1, 3 * k) -
                                                    np.exp(-2 * k) * iv(1, k))

        # compute the half-height
        half_height = (1 / 2) * pi * k * r0 * (iv(1, k) + modstruve(-1, k))
        outputs["half-height"] = half_height

    def setup_partials(self):
        size = self._get_var_meta("θ", "size")
        self.declare_partials("k", ["Ib TF R_out", "ΔR"], method="exact")
        self.declare_partials("d_sq", ["Ib TF R_out"], method="fd")
        self.declare_partials("d_sq", ["R0", "ΔR"], method="fd")
        self.declare_partials("d_sq", ["θ"],
                              method="exact",
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("Ob TF R_in", ["ΔR", "Ib TF R_out"], val=1)
        self.declare_partials("constraint_axis_within_coils",
                              ["ΔR", "Ib TF R_out"],
                              val=1)
        self.declare_partials("constraint_axis_within_coils", ["R0"], val=-1)
        self.declare_partials("V_enc", ["Ib TF R_out", "ΔR"], method="exact")
        self.declare_partials("half-height", ["Ib TF R_out", "ΔR"],
                              method="exact")

    def compute_partials(self, inputs, J):
        size = self._get_var_meta("θ", "size")
        r_ot = inputs["Ib TF R_out"]
        r1 = r_ot
        ΔR = inputs["ΔR"]
        R0 = inputs["R0"]
        r2 = r_ot + ΔR
        k = np.log(r2 / r1) / 2
        r0 = np.sqrt(r1 * r2)
        J["k", "Ib TF R_out"] = -ΔR / (2 * r_ot * r2)
        J["k", "ΔR"] = 1 / (2 * r2)

        # angle of the transition between the vertical inner leg and the curve
        θ_all = inputs["θ"]
        θ_max_curve = self.terminal_theta(k, r0, r1, R0)

        θ = θ_all
        on_upper_curve = (θ >= 0) * (θ < θ_max_curve)
        on_lower_curve = (θ < 0) * (θ > -θ_max_curve)
        on_straight = np.logical_or(θ <= -θ_max_curve, θ >= θ_max_curve)

        θ = θ_all[on_straight]

        dρ2dθ = np.zeros(size)

        dρ2dθ[on_straight] = 2 * (R0 - r_ot)**2 * np.cos(θ)**(-2) * np.tan(θ)

        θ = θ_all[on_upper_curve]
        sols = self.solve_curve(r1, r2, R0, θ)
        dρ2dθ[on_upper_curve] = 2 * sols["ρ"] * sols["dρ/dθ"]

        θ = θ_all[on_lower_curve]
        sols = self.solve_curve(r1, r2, R0, -θ[::-1])
        dρ2dθ[on_lower_curve] = -2 * sols["ρ"][::-1] * sols["dρ/dθ"][::-1]

        J["d_sq", "θ"] = dρ2dθ
        dV_dk = 2 * np.exp(-2 * k) * k * pi**2 * r0**3 * (
            -iv(0, k) + 3 * np.exp(2 * k) * iv(0, 3 * k) + 2 * iv(1, k))
        dV_dr0 = 6 * k * pi**2 * r0**2 * (iv(1, 3 * k) -
                                          np.exp(-2 * k) * iv(1, k))

        dr0_drot = (2 * r_ot + ΔR) / (2 * (r_ot * (r_ot + ΔR))**(1 / 2))
        dr0_dΔR = r_ot / (2 * (r_ot * (r_ot + ΔR))**(1 / 2))
        J["V_enc", "ΔR"] = dV_dk * J["k", "ΔR"] + dV_dr0 * dr0_dΔR
        J["V_enc",
          "Ib TF R_out"] = dV_dk * J["k", "Ib TF R_out"] + dV_dr0 * dr0_drot

        dhh_dr0 = (1 / 2) * pi * k * (iv(1, k) + modstruve(-1, k))
        dhh_dk = ((1 / 2) * pi * k * r0 * (iv(0, k) + modstruve(-2, k)) +
                  pi * r0 * modstruve(-1, k))
        J["half-height", "ΔR"] = dhh_dk * J["k", "ΔR"] + dhh_dr0 * dr0_dΔR
        J["half-height",
          "Ib TF R_out"] = dhh_dk * J["k", "Ib TF R_out"] + dhh_dr0 * dr0_drot

    def plot(self, ax=None, **kwargs):
        color = "black"

        if "color" in kwargs.keys():
            color = kwargs[color]

        θ = self.get_val("θ")
        R0 = self.get_val("R0")
        ρ = self.get_val("d_sq")**(1 / 2)
        c_R = R0 + ρ * np.cos(θ)
        c_Z = ρ * np.sin(θ)
        ax.plot(c_R, c_Z, color=color, **kwargs)


if __name__ == "__main__":
    prob = om.Problem()

    θ = np.linspace(-pi, pi, 31, endpoint=True)

    prob.model.add_subsystem("ivc",
                             om.IndepVarComp("θ", val=θ),
                             promotes_outputs=["*"])

    pds = PrincetonDeeTFSet()
    prob.model.add_subsystem("pdTF", pds, promotes_inputs=["*"])

    prob.setup()

    # sol = pds.solve_curve(1, 2, 1.5, np.array([0, np.pi/2]))
    # print(sol)
    prob.set_val("Ib TF R_out", 1.0)
    prob.set_val("R0", 1.5)
    prob.set_val("ΔR", 1)
    # prob.set_val("θ", [1,2,3])

    prob.run_driver()
    # all_inputs = prob.model.list_inputs(print_arrays=True)
    all_outputs = prob.model.list_outputs(print_arrays=True)
