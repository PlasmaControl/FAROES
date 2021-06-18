import numpy as np
from math import pi as π
import scipy as scipy
from scipy.special import ellipe, hyp2f1
from numpy import sin, cos

import openmdao.api as om
from openmdao.utils.cs_safe import abs as cs_safe_abs
from openmdao.utils.cs_safe import arctan2 as cs_safe_arctan2

from plasmapy.particles import Particle, common_isotopes
from plasmapy.particles import atomic_number, isotopic_abundance


class DoubleSmoothShiftedReLu(om.Group):
    r"""Three sloped regions, with smooth transitions.

    The left region has value and slope of 0,
    The center region has one slope,
    and the right region has another slope.

    .. image :: images/smooth_shifted_double_relu.png

    Inputs
    ------
    x : float

    Outputs:
    y : float

    Notes:
    ------
    Config options:
    sharpness : float
       sharpness of curve. typically >10
    x0 : float
       Point at which it turns up
    s1 : float
       First slope
    x1 : float
       Point at which it turns again. Must be larger than x0.
    s2 : float
       Second slope
    """
    def initialize(self):
        self.options.declare("sharpness", default=10)
        self.options.declare("x0", default=0)
        self.options.declare("x1", default=1)
        self.options.declare("s1", default=1)
        self.options.declare("s2", default=0)
        self.options.declare("units_out", default=None)

    def setup(self):
        b = self.options["sharpness"]
        x0 = self.options["x0"]
        x1 = self.options["x1"]
        s1 = self.options["s1"]
        s2 = self.options["s2"]
        u_o = self.options["units_out"]
        diff = s2 - s1

        one = SmoothShiftedReLu(bignum=b, x0=x0)
        two = SmoothShiftedReLu(bignum=b, x0=x1)
        self.add_subsystem("one",
                           one,
                           promotes_inputs=["x"],
                           promotes_outputs=[("y", "oney")])
        self.add_subsystem("two",
                           two,
                           promotes_inputs=["x"],
                           promotes_outputs=[("y", "twoy")])
        self.add_subsystem("out",
                           om.ExecComp(f"y = {s1} * oney + {diff} * twoy",
                                       y={"units": u_o}),
                           promotes_inputs=["*"],
                           promotes_outputs=["y"])


class SmoothShiftedReLu(om.ExplicitComponent):
    r"""
    Inputs
    ------
    x : float

    Outputs
    -------
    y : float

    Notes
    -----
    config options:
    bignum : float
       sharpness of curve. typically >10
    x0 : float
       Point at which it starts to turn on
    """
    def initialize(self):
        self.options.declare('x0', default=0)
        self.options.declare('bignum', default=10)

    def setup(self):
        self.b = self.options["bignum"]
        self.x0 = self.options["x0"]
        self.add_input("x")
        self.add_output("y")

    def compute(self, inputs, outputs):
        b = self.b
        x0 = self.x0
        x = inputs["x"]
        y = (1 / b) * np.log(1 + np.exp(b * (x - x0)))
        outputs["y"] = y

    def setup_partials(self):
        self.declare_partials("y", "x")

    def compute_partials(self, inputs, J):
        b = self.b
        x0 = self.x0
        x = inputs["x"]
        J["y", "x"] = 1 / (1 + np.exp(b * (x0 - x)))


class PolarAngleAndDistanceFromPoint(om.ExplicitComponent):
    r"""
    Inputs
    ------
    x : array
        m, x-locations of points
    y : array
        m, y-locations of points

    X0 : float
        m, Polar origin x
    Y0 : float
        m, Polar origin y

    Outputs
    -------
    d_sq : array
        m**2, Squared distances from origin to points (x,y)
    θ : array
        Angle from origin to points (x,y).
           In the range (-pi, pi].
    """
    def setup(self):
        self.add_input("x", units="m", shape_by_conn=True)
        self.add_input("y", units="m", shape_by_conn=True, copy_shape="x")
        self.add_input("X0", units="m")
        self.add_input("Y0", units="m", val=0)

        self.add_output("d_sq",
                        units="m**2",
                        lower=0,
                        ref=10,
                        shape_by_conn=True,
                        copy_shape="x")
        self.add_output("θ",
                        lower=-π,
                        upper=π,
                        shape_by_conn=True,
                        copy_shape="x")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        X0 = inputs["X0"]
        Y0 = inputs["Y0"]

        d_sq = (x - X0)**2 + (y - Y0)**2
        θ = cs_safe_arctan2(y - Y0, x - X0)

        outputs["d_sq"] = d_sq
        outputs["θ"] = θ

    def setup_partials(self):
        size = self._get_var_meta("x", "size")
        self.declare_partials(["d_sq", "θ"], ["x", "y"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials(["d_sq", "θ"], ["X0", "Y0"])

    def compute_partials(self, inputs, J):
        x = inputs["x"]
        y = inputs["y"]
        X0 = inputs["X0"]
        Y0 = inputs["Y0"]
        d_sq = (x - X0)**2 + (y - Y0)**2
        J["d_sq", "x"] = 2 * (x - X0)
        J["d_sq", "X0"] = -2 * (x - X0)
        J["d_sq", "y"] = 2 * (y - Y0)
        J["d_sq", "Y0"] = -2 * (y - Y0)
        J["θ", "x"] = -(y - Y0) / d_sq
        J["θ", "y"] = (x - X0) / d_sq
        J["θ", "X0"] = -J["θ", "x"]
        J["θ", "Y0"] = -J["θ", "y"]


class OffsetParametricCurvePoints(om.ExplicitComponent):
    r"""
    Inputs
    ------
    x : array
        m, x-locations of points on a parametric curve
    y : array
        m, y-locations of points on a parametric curve
    dx_dt : array
        m, Derivative of x location of each point w.r.t. the curve parameter t
    dy_dt : array
        m, Derivative of y location of each point w.r.t. the curve parameter t
    s : float
        m, Perpendicular offset of the resulting curve from the original.

    Outputs
    -------
    x_o : array
        m, x-locations of points on offset curve
    y_o : array
        m, y-locations of points on offset curve
    θ_o : array
        Angle of the offset distance

    References
    ----------
    .. [1] https://mathworld.wolfram.com/ParallelCurves.html
    .. [2] https://en.wikipedia.org/wiki/Parallel_curve
    """
    def setup(self):
        self.add_input("x", units="m", shape_by_conn=True)
        self.add_input("y", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("dx_dt", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("dy_dt", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("s", units="m", desc="offset")

        self.add_output("x_o", units="m", copy_shape="x")
        self.add_output("y_o", units="m", copy_shape="x")
        self.add_output("θ_o", copy_shape="x")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        dx_dt = inputs["dx_dt"]
        dy_dt = inputs["dy_dt"]
        s = inputs["s"]
        x_o = x + s * dy_dt / (dx_dt**2 + dy_dt**2)**(1 / 2)
        y_o = y - s * dx_dt / (dx_dt**2 + dy_dt**2)**(1 / 2)
        outputs["x_o"] = x_o
        outputs["y_o"] = y_o
        outputs["θ_o"] = cs_safe_arctan2(y=-dx_dt, x=dy_dt)

    def setup_partials(self):
        size = self._get_var_meta("x", "size")
        self.declare_partials("x_o", ["x"], rows=range(size), cols=range(size))
        self.declare_partials("x_o", ["dx_dt"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("x_o", ["dy_dt"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("x_o", ["s"], val=np.zeros(size))
        self.declare_partials("y_o", ["y"], rows=range(size), cols=range(size))
        self.declare_partials("y_o", ["dx_dt"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("y_o", ["dy_dt"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("y_o", ["s"], val=np.zeros(size))
        self.declare_partials("θ_o", ["dx_dt", "dy_dt"],
                              rows=range(size),
                              cols=range(size))

    def compute_partials(self, inputs, J):
        size = self._get_var_meta("x", "size")
        J["x_o", "x"] = np.ones(size)
        J["y_o", "y"] = np.ones(size)

        s = inputs["s"]
        dx_dt = inputs["dx_dt"]
        dy_dt = inputs["dy_dt"]
        denom12 = (dx_dt**2 + dy_dt**2)**(1 / 2)
        dxo_ds = dy_dt / denom12
        J["x_o", "s"] = dxo_ds
        dyo_ds = -dx_dt / denom12
        J["y_o", "s"] = dyo_ds

        denom32 = (dx_dt**2 + dy_dt**2)**(3 / 2)
        dxo_dxdt = -s * dx_dt * dy_dt / denom32
        J["x_o", "dx_dt"] = dxo_dxdt

        dxo_dydt = s * dx_dt**2 / denom32
        J["x_o", "dy_dt"] = dxo_dydt

        dyo_dydt = s * dx_dt * dy_dt / denom32
        J["y_o", "dy_dt"] = dyo_dydt

        dyo_dxdt = -s * dy_dt**2 / denom32
        J["y_o", "dx_dt"] = dyo_dxdt

        J["θ_o", "dx_dt"] = -dy_dt / (dx_dt**2 + dy_dt**2)
        J["θ_o", "dy_dt"] = dx_dt / (dx_dt**2 + dy_dt**2)

    def plot(self, ax=None, **kwargs):
        x = self.get_val('x_o')
        y = self.get_val('y_o')
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ax.plot(x, y, **kwargs)


class OffsetCurveWithLimiter(om.ExplicitComponent):
    r"""

    Inputs
    ------
    x : array
        m, x-locations of points on a parametric curve
    y : array
        m, y-locations of points on a parametric curve
    θ_o : array
        Angle between the original point and the offset point
    s : float
        m, Perpendicular offset of the resulting curve from the original.
    x_min : float
        m, Minimum x for offset points. Points that would otherwise exceed this
           will have the local s decreased. Must be equal to or smaller than
           all points in x.

    Outputs
    -------
    x_o : array
        m, x-locations of points on offset curve
    y_o : array
        m, y-locations of points on offset curve

    Notes
    -----
    All points x must be greater than or equal to xmin.

    References
    ----------
    .. [1] Cook, John D.
       www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
    """
    def setup(self):
        self.add_input("x", units="m", shape_by_conn=True)
        self.add_input("y", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("θ_o", copy_shape="x", shape_by_conn=True)
        self.add_input("s", units="m", desc="offset")
        self.add_input("x_min", units="m", val=0.0)

        self.add_output("x_o", units="m", copy_shape="x")
        self.add_output("y_o", units="m", copy_shape="x")
        self.b = 30
        self.θε = 1e-3  # prevents dealing with tan(π/2) = infinity

    def compute(self, inputs, outputs):
        b = self.b
        size = self._get_var_meta("x", "size")
        x = inputs["x"]
        y = inputs["y"]
        θ_o = inputs["θ_o"]
        s = inputs["s"]
        x_min = inputs["x_min"]

        if self.under_complex_step:
            array_type = np.cdouble
        else:
            array_type = np.double
        x_o = np.zeros(size, dtype=array_type)
        y_o = np.zeros(size, dtype=array_type)

        # the inboard side may be modified; the outboard side will be left
        # alone.
        ib = np.logical_or(θ_o > np.pi / 2 + self.θε,
                           θ_o < -np.pi / 2 - self.θε)

        # initial computation for x_o, y_o; the ones on the inner half
        # will be rewritten
        x_o = x + s * np.cos(θ_o)
        y_o = y + s * np.sin(θ_o)

        # construct a new x_o.
        case1 = np.logical_and(x_o >= x_min, ib)
        case2 = np.logical_and(x_o < x_min, ib)

        xo1 = x_o[case1]
        xo2 = x_o[case2]

        xo1_new = xo1 + np.log(1 + np.exp(b * (x_min - xo1))) / b
        xo2_new = x_min + np.log(1 + np.exp(b * (xo2 - x_min))) / b

        x_o[case1] = xo1_new
        x_o[case2] = xo2_new

        # use that to find s_new for that point
        # Since π/2 < θ < 3 π/2 cosine will not be zero.
        y_o[ib] = y[ib] + (x_o[ib] - x[ib]) * np.tan(θ_o[ib])

        outputs["x_o"] = x_o
        outputs["y_o"] = y_o

    def setup_partials(self):
        # perhaps this can be written in a more compact manner
        size = self._get_var_meta("x", "size")
        self.declare_partials("x_o", ["x"], rows=range(size), cols=range(size))
        self.declare_partials("x_o", ["θ_o"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("x_o", ["s"], val=np.zeros(size))
        self.declare_partials("x_o", ["x_min"], val=np.zeros(size))

        self.declare_partials("y_o", ["x"], rows=range(size), cols=range(size))
        self.declare_partials("y_o", ["y"], rows=range(size), cols=range(size))
        self.declare_partials("y_o", ["θ_o"],
                              rows=range(size),
                              cols=range(size))
        self.declare_partials("y_o", ["s"], val=np.zeros(size))
        self.declare_partials("y_o", ["x_min"], val=np.zeros(size))

    def compute_partials(self, inputs, J):
        b = self.b
        size = self._get_var_meta("x", "size")
        x = inputs["x"]
        θ_o = inputs["θ_o"]
        s = inputs["s"]
        x_min = inputs["x_min"]

        # basic offset
        dxo_dx = np.ones(size)
        dxo_dθ = -s * np.sin(θ_o)
        dxo_ds = np.cos(θ_o)
        dxo_dxmin = np.zeros(size)

        # now apply the minimum x computation
        ib = np.logical_or(θ_o > np.pi / 2 + self.θε,
                           θ_o < -np.pi / 2 - self.θε)
        x_o = x + s * np.cos(θ_o)
        case1 = np.logical_and(x_o >= x_min, ib)
        case2 = np.logical_and(x_o < x_min, ib)
        xo1 = x_o[case1]
        xo2 = x_o[case2]

        dxo1n_dxmin = (np.exp(b * (x_min - xo1)) / (1 + np.exp(b *
                                                               (x_min - xo1))))
        dxo1n_dxo1 = 1 - dxo1n_dxmin
        dxo2n_dxo2 = (np.exp(b * (xo2 - x_min)) / (1 + np.exp(b *
                                                              (xo2 - x_min))))
        dxo2n_dxmin = 1 - dxo2n_dxo2

        dxo_dx[case1] = dxo1n_dxo1 * dxo_dx[case1]
        dxo_dx[case2] = dxo2n_dxo2 * dxo_dx[case2]
        dxo_dθ[case1] = dxo1n_dxo1 * dxo_dθ[case1]
        dxo_dθ[case2] = dxo2n_dxo2 * dxo_dθ[case2]
        dxo_ds[case1] = dxo1n_dxo1 * dxo_ds[case1]
        dxo_ds[case2] = dxo2n_dxo2 * dxo_ds[case2]

        dxo_dxmin[case1] = dxo1n_dxmin
        dxo_dxmin[case2] = dxo2n_dxmin

        J["x_o", "x"] = dxo_dx
        J["x_o", "θ_o"] = dxo_dθ
        J["x_o", "s"] = dxo_ds
        J["x_o", "x_min"] = dxo_dxmin

        dyo_dx = np.zeros(size)
        dyo_dy = np.ones(size)
        dyo_dθ = s * np.cos(θ_o)
        dyo_ds = np.sin(θ_o)
        dyo_dxmin = np.zeros(size)

        xo1 = x_o[case1]
        xo2 = x_o[case2]

        x_o[case1] = xo1 + np.log(1 + np.exp(b * (x_min - xo1))) / b
        x_o[case2] = x_min + np.log(1 + np.exp(b * (xo2 - x_min))) / b

        dyo_dx[ib] = (dxo_dx[ib] - 1) * np.tan(θ_o[ib])
        dyo_dθ[ib] = (dxo_dθ[ib] * np.tan(θ_o[ib]) +
                      (x_o[ib] - x[ib]) / np.cos(θ_o[ib])**2)
        dyo_ds[ib] = dxo_ds[ib] * np.tan(θ_o[ib])
        dyo_dxmin[ib] = dxo_dxmin[ib] * np.tan(θ_o[ib])

        J["y_o", "x"] = dyo_dx
        J["y_o", "y"] = dyo_dy
        J["y_o", "θ_o"] = dyo_dθ
        J["y_o", "s"] = dyo_ds
        J["y_o", "x_min"] = dyo_dxmin

    def plot(self, ax=None, **kwargs):
        x = self.get_val('x_o')
        y = self.get_val('y_o')
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ax.plot(x, y, **kwargs)


class PolarParallelCurve(om.Group):
    r"""Builds a parallel curve of constant width
    and returns the polar locations of points on the curve

    Inputs
    ------
    R : array
       m, Radial location of initial curve points
    Z : array
       m, Z location of initial curve points
    dR_dθ : array
       m, dependence of R points on curve parameter θ
    dR_dZ : array
       m, dependence of Z points on curve parameter θ
    s : float
       m, Parallel offset distance
    R0 : float
       m, Radial coordinate of polar origin
    Z0 : float
       m, Z coordinate of polar origin

    Outputs
    -------
    d_sq : array
       m**2, squared distances to points on the parallel curve
    θ_parall : array
       Angles to points on the parallel curve
    """
    def initialize(self):
        self.options.declare('use_Rmin', default=False)
        self.options.declare('torus_V', default=False)

    def setup(self):
        use_Rmin = self.options['use_Rmin']
        torus_V = self.options['torus_V']
        self.add_subsystem("pts",
                           OffsetParametricCurvePoints(),
                           promotes_inputs=[("x", "R"), ("y", "Z"), "s",
                                            ("dx_dt", "dR_dθ"),
                                            ("dy_dt", "dZ_dθ")])

        if use_Rmin:
            self.add_subsystem("limiter",
                               OffsetCurveWithLimiter(),
                               promotes_inputs=[("x", "R"), ("y", "Z"), "s",
                                                ("x_min", "R_min")])
            self.connect("pts.θ_o", "limiter.θ_o")
            points_comp = "limiter"
        else:
            points_comp = "pts"

        self.add_subsystem("d_sq_theta",
                           PolarAngleAndDistanceFromPoint(),
                           promotes_inputs=[("X0", "R0"), ("Y0", "Z0")],
                           promotes_outputs=["d_sq", ("θ", "θ_parall")])
        self.connect(points_comp + ".x_o", "d_sq_theta.x")
        self.connect(points_comp + ".y_o", "d_sq_theta.y")

        if torus_V:
            self.add_subsystem("V_enc",
                               PolygonalTorusVolume(),
                               promotes_outputs=["V"])
            self.connect(points_comp + ".x_o", "V_enc.R")
            self.connect(points_comp + ".y_o", "V_enc.Z")


class SoftCapUnity(om.ExplicitComponent):
    r"""Limits functions that can go a bit over 1 to 1

    .. math::

       1 - \frac{1}{b}\left(\log(1 + \exp(b(1 - x)))\right)

    where :math:`b` is a parameter; typically 20.

    Inputs
    ------
    x : float
        Input; should be 0 to 1.3

    Outputs
    -------
    y : float
        Output; range is 0 to 1
    """
    def initialize(self):
        self.options.declare('b', default=20)

    def setup(self):
        self.b = self.options['b']
        self.add_input("x")
        self.add_output("y", lower=0, val=0.9)

    def compute(self, inputs, outputs):
        b = self.b
        x = inputs["x"]
        y = 1 - (1 / b) * np.log(1 + np.exp(b * (1 - x)))
        outputs["y"] = y

    def setup_partials(self):
        self.declare_partials('y', 'x')

    def compute_partials(self, inputs, J):
        b = self.b
        x = inputs["x"]
        J['y', 'x'] = np.exp(b) / (np.exp(b) + np.exp(b * x))


class Softmax(om.ExplicitComponent):
    r"""Soft maximum of an array x and a value y

    .. math::

       z = \max(x,y) - \frac{1}{b}
           \left(\log(1 + \exp(b(\min(x,y) - \max(x,y))))\right)

    where :math:`b` is a parameter; typically 20.

    Inputs
    ------
    x : array
        An array of floats
    y : float
        A single value

    Outputs
    -------
    z : array
        An array of floats

    References
    ----------
    https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
    """
    def initialize(self):
        self.options.declare('b', default=20)
        self.options.declare('units', default='m')

    def setup(self):
        self.b = self.options['b']
        u = self.options['units']
        self.add_input("x", shape_by_conn=True, units=u)
        self.add_input("y", units=u)
        self.add_output("z", copy_shape="x", units=u)

    def compute(self, inputs, outputs):
        b = self.b
        x = inputs["x"]
        y = inputs["y"]
        size = self._get_var_meta("x", "size")
        z = np.zeros(size, dtype=np.cdouble)

        case1 = x >= y
        case2 = x < y

        x1 = x[case1]
        x2 = x[case2]

        z[case1] = x1 + np.log(1 + np.exp(b * (y - x1))) / b
        z[case2] = y + np.log(1 + np.exp(b * (x2 - y))) / b
        outputs["z"] = z

    def setup_partials(self):
        size = self._get_var_meta("x", "size")
        self.declare_partials('z', ['x'], rows=range(size), cols=range(size))
        self.declare_partials('z', ['y'])

    def compute_partials(self, inputs, J):
        size = self._get_var_meta("x", "size")
        dz_dx = np.zeros(size, dtype=np.double)
        dz_dy = np.zeros(size, dtype=np.double)

        b = self.b
        x = inputs["x"]
        y = inputs["y"]

        case1 = x >= y
        case2 = x < y
        x1 = x[case1]
        x2 = x[case2]

        dz_dx[case1] = 1 - np.exp(b * (y - x1)) / (1 + np.exp(b * (y - x1)))
        dz_dy[case1] = np.exp(b * (y - x1)) / (1 + np.exp(b * (y - x1)))

        dz_dx[case2] = np.exp(b * (x2 - y)) / (1 + np.exp(b * (x2 - y)))
        dz_dy[case2] = 1 - np.exp(b * (x2 - y)) / (1 + np.exp(b * (x2 - y)))

        J['z', 'x'] = dz_dx
        J['z', 'y'] = dz_dy


class PowerScalingLaw(om.ExplicitComponent):
    r"""Configurable power scaling law

    .. math::

       \mathrm{out} / u_\mathrm{out} = c_0 (a/u_a)^{c_a} (b/u_b)^{c_b} \ldots

    where :math:`c_0` is an initial constant, :math:`a, b` are variables,
    :math:`u_a, u_b` are units of their respective variables, and
    :math:`c_a, c_b` are power law exponents for their respective variables.

    Input variables must be non-negative.
    """
    NEGATIVE_TERM = "Term '%s' is non-positive in " + \
        "the scaling law calculation. Its value was %f."

    def initialize(self):
        self.options.declare("const", default="c0", types=str)
        self.options.declare("output", default="out", types=str)
        self.options.declare("ref", default=1.0, types=float)
        self.options.declare("lower", default=0.0, types=float)
        self.options.declare("terms", default=None, types=dict)
        self.options.declare("term_units", default=None, types=dict)

    def setup(self):
        self.CONST = self.options["const"]
        self.OUTPUT = self.options["output"]
        ref = self.options["ref"]
        lower = self.options["lower"]
        terms = self.options["terms"].copy()
        term_units = self.options["term_units"].copy()

        for k in terms.keys():
            if k != self.CONST:
                self.add_input(k, units=term_units[k])

        self.constant = terms.pop(self.CONST)
        self.varterms = terms

        self.add_output(self.OUTPUT,
                        lower=lower,
                        units=term_units[self.CONST],
                        ref=ref)

    def compute(self, inputs, outputs):
        out = self.constant

        for k, v in self.varterms.items():
            term = inputs[k]
            if term <= 0:
                raise om.AnalysisError(self.NEGATIVE_TERM % (k, term))
            out *= term**v

        outputs[self.OUTPUT] = out

    def setup_partials(self):
        for k in self.varterms.keys():
            self.declare_partials(self.OUTPUT, k)

    def partial(self, inputs, J, var):
        doutdv = self.constant
        for k, v in self.varterms.items():
            if k != var:
                doutdv *= inputs[k]**v
            else:
                doutdv *= v * inputs[k]**(v - 1)
        J[self.OUTPUT, var] = doutdv

    def compute_partials(self, inputs, J):
        for k in self.varterms.keys():
            self.partial(inputs, J, k)


class PolygonalTorusVolume(om.ExplicitComponent):
    r"""A torus specified by (R, Z) points

    Inputs
    ------
    R : array
       m, Radial locations
    Z : array
       m, Vertical locations

    Outputs
    -------
    V : float
       m**3, Enclosed volume
    """
    def setup(self):
        self.add_input("R", units="m", shape_by_conn=True)
        self.add_input("Z", units="m", copy_shape="R", shape_by_conn=True)
        self.add_output("V", units="m**3", ref=100)

    def compute(self, inputs, outputs):
        r = inputs["R"]
        z = inputs["Z"]
        ra = np.append(r, r[0])
        za = np.append(z, z[0])
        r0 = ra[0:-1]
        z0 = za[0:-1]
        r1 = ra[1:]
        z1 = za[1:]
        Vs = (π / 3) * (r0 - r1) * (r0 * (2 * z0 + z1) + r1 * (z0 + 2 * z1))
        outputs["V"] = np.sum(Vs)

    def setup_partials(self):
        size = self._get_var_meta("R", "size")
        self.declare_partials("V", "R", val=np.zeros(size))
        self.declare_partials("V", "Z", val=np.zeros(size))

    def compute_partials(self, inputs, J):
        r = inputs["R"]
        z = inputs["Z"]
        ra = np.append(r, r[0])
        za = np.append(z, z[0])
        r0 = ra[0:-1]
        z0 = za[0:-1]
        r1 = ra[1:]
        z1 = za[1:]
        dV_dr0 = (π / 3) * (r1 * (z1 - z0) + 2 * r0 * (2 * z0 + z1))
        dV_dr1 = -(π / 3) * (r0 * (z0 - z1) + 2 * r1 * (z0 + 2 * z1))
        dV_dz0 = (π / 3) * (r0 - r1) * (2 * r0 + r1)
        dV_dz1 = (π / 3) * (r0 - r1) * (r0 + 2 * r1)
        J["V", "R"] = dV_dr0 + np.roll(dV_dr1, 1)
        J["V", "Z"] = dV_dz0 + np.roll(dV_dz1, 1)


def most_common_isotope(sp):
    """A Particle of the most common isotope and
    maximum charge for the given species.

    Parameters
    ----------
    sp : str
       Element name or symbol of the species

    Returns
    -------
    Particle
    """
    isotopes = common_isotopes(sp)
    max_charge = atomic_number(sp)
    abundances = []
    for i in isotopes:
        abundances.append(isotopic_abundance(i))
    isotope_index = np.argmax(abundances)
    most_common_isotope = isotopes[isotope_index]
    mass_number = Particle(most_common_isotope).mass_number
    impurity = Particle(max_charge, Z=max_charge, mass_numb=mass_number)
    return impurity


def tube_segment_volume(r_i, r_o, h):
    """Volume of a finite tube's wall
    """
    V = π * (r_o**2 - r_i**2) * h
    return V


def tube_segment_volume_derivatives(r_i, r_o, h):
    """Derivatives for volume of a finite tube's wall
    """
    dVdr_i = -2 * π * h * r_i
    dVdr_o = 2 * π * h * r_o
    dVdh = π * (r_o**2 - r_i**2)
    return {'r_i': dVdr_i, 'r_o': dVdr_o, 'h': dVdh}


def ellipse_perimeter_simple(a, b):
    """Often seen as √((1 + κ^2)/2)

    Parameters
    ----------
    a : float
       short minor radius of an ellipse
    b : float
       long minor radius of an ellipse

    Returns
    -------
    P : float
       perimeter of the ellipse
    """
    P = 2 * π * a * np.sqrt((1 + (b / a)**2) / 2)
    return P


def ellipse_perimeter_simple_derivatives(a, b):
    """Often seen as √((1 + κ^2)/2)

    Parameters
    ----------
    a : float
       short minor radius of an ellipse
    b : float
       long minor radius of an ellipse

    Returns
    -------
    Dict of
    a : float
       derivative with respect to a
    b : float
       derivative with respect to b
    """
    dP_da = 2**(1 / 2) * π / (1 + (b / a)**2)**(1 / 2)
    dP_db = 2**(1 / 2) * (b / a) * π / (1 + (b / a)**2)**(1 / 2)
    return {"a": dP_da, "b": dP_db}


def ellipse_perimeter(a, b):
    """Exact formula using special functions

    Parameters
    ----------
    a : float
       one minor radius of an ellipse
    b : float
       other minor radius of an ellipse

    Returns
    -------
    P : float
       perimeter of the ellipse
    """
    return 4 * b * ellipe(1 - a**2 / b**2)


def ellipse_perimeter_derivatives(a, b):
    r"""Exact formula using special functions

    Parameters
    ----------
    a : float
       one minor radius of an ellipse
    b : float
       other minor radius of an ellipse

    Returns
    -------
    Dict of
    a : float
       derivative with respect to a
    b : float
       derivative with respect to b

    Notes
    -----
    We use a different form here to avoid zeros in denominators when a==b

    .. code-block:: none

        D[\[Pi] Sqrt[2 (a^2 + b^2)]
        Hypergeometric2F1[-(1/4), 1/4, 1, (a^2 - b^2)^2/(a^2 + b^2)^2], a]

    References
    ----------
    https://www.mathematica-journal.com/2009/11/23/on-the-perimeter-of-an-ellipse/
    """
    def epd(a, b):
        hyp1 = hyp2f1(-1 / 4, 1 / 4, 1, ((a**2 - b**2) / (a**2 + b**2))**2)
        num1 = 2**(1 / 2) * a * π * hyp1
        den1 = (a**2 + b**2)**(1 / 2)
        term1 = num1 / den1

        hyp2 = hyp2f1(3 / 4, 5 / 4, 2, ((a**2 - b**2) / (a**2 + b**2))**2)
        num2 = 8 * a * (a - b) * b**2 * (a + b) / (a**2 + b**2)**3 * π
        num2 = num2 * hyp2 * (a**2 + b**2)**(1 / 2)
        den2 = 8 * 2**(1 / 2)
        term2 = num2 / den2
        return term1 - term2

    dPda = epd(a, b)
    dPdb = epd(b, a)  # formula is symmetric
    return {'a': dPda, 'b': dPdb}


def ellipse_perimeter_ramanujan(a, b):
    """Surprisingly accurate formula for ellipse perimeter

    Parameters
    ----------
    a : float
       one semi-axis of an ellipse
    b : float
       other semi-axis of an ellipse

    Returns
    -------
    P : float
       perimeter of the ellipse
    """
    P = π * (3 * (a + b) - np.sqrt((3 * a + b) * (3 * b + a)))
    return P


def ellipse_perimeter_ramanujan_derivatives(a, b):
    """Partial derivatives for ellipse_perimeter_ramanujan
    """
    dPda = π * (3 - (3 * a + 5 * b) / np.sqrt((3 * a + b) * (a + 3 * b)))
    dPdb = π * (3 - (5 * a + 3 * b) / np.sqrt((3 * a + b) * (a + 3 * b)))
    return {'a': dPda, 'b': dPdb}


def polar_offset_ellipse(a, b, x, y, t):
    r"""Radius to an offset ellipse

    The distance in polar form to an ellipse which is
    offset from the origin is [1]_

    .. math::

       \rho(\theta) = \frac{b^2 x \cos (\theta)+a^2 y \sin (\theta)+a b
           \sqrt{\left(b^2-y^2\right) \cos ^2(\theta)+2 x y \cos (\theta)
           \sin(\theta)+\left(a^2-x^2\right) \sin ^2(\theta)}}
           {b^2 \cos ^2(\theta)+a^2 \sin ^2(\theta)}

    where :math:`a, b` are the horizontal and vertical semi-axes of the
    ellipse, respectively, and the center of the ellipse is at :math:`(x,y)`.

    Parameters
    ----------
    a : float
       horizontal semi-axis of the ellipse
    b : float
       vertical semi-axis of the ellipse
    x : float
       horizontal ellipse center location
    y : float
       vertical ellipse center location
    t : float
       polar angle

    Returns
    -------
    radius : float

    References
    ----------
    .. [1] http://www.jaschwartz.net/journal/offset-ellipse-polar-form.html
    """
    s = sin(t)
    c = cos(t)

    root = np.sqrt((b**2 - y**2) * c**2 + 2 * x * y * c * s +
                   (a**2 - x**2) * s**2)
    numer = b**2 * x * c + a**2 * y * s + a * b * root
    denom = (b * c)**2 + (a * s)**2
    d = numer / denom
    return d


def torus_surface_area(R, a, b=None):
    """Simple formula for surface area of a (elliptical) torus

    Parameters
    ----------
    R : float
       major radius
    a : float
       horizontal minor radius
    b : float [optional]
       vertical minor radius

    Returns
    -------
    sa : float
        Surface area
    """
    if b is not None:
        circumference = ellipse_perimeter_ramanujan(a, b)
    else:
        circumference = 2 * π * a
    sa = 2 * π * R * circumference
    return sa


def torus_volume(R, a, b=None):
    """Volume of a (elliptical) torus

    Parameters
    ----------
    R : float
       major radius
    a : float
       horizontal minor radius
    b : float [optional]
       vertical minor radius

    Returns
    -------
    V : float
        Volume
    """
    if b is None:
        b = a
    area = π * a * b
    circum = 2 * π * R
    V = area * circum
    return V


def half_ellipse_torus_volume(R, a, b):
    """Volume of a torus with a cross section
    shaped like half of an ellipse (vertical slice)

    Parameters
    ----------
    R : float
        major radius of the torus (radius to flat side of the half-ellipse)
    a : horizontal semi-axis of the ellipse.
        positive values: shape like | ◗
        negative values: shape like | ◖
        where | is the centerline
    b : vertical semi-axis of the ellipse.
    """
    return cs_safe_abs(a) * b * π * (4 * a + 3 * π * R) / 3


def half_ellipse_torus_volume_derivatives(R, a, b):
    """Derivatives of volume of a torus with a cross section
    shaped like half of an ellipse (vertical slice)

    Parameters
    ----------
    R : float
        major radius of the torus (radius to flat side of the half-ellipse)
    a : horizontal semi-axis of the ellipse.
        positive values: shape like | ◗
        negative values: shape like | ◖
        where | is the centerline
    b : vertical semi-axis of the ellipse.

    Returns
    -------
    Dict of
    R : float
       derivative with respect to R
    a : float
       derivative with respect to a
    b : float
       derivative with respect to b

    Notes
    -----
    May not be well-defined at a=0. This is the degenerate case.
    """
    dVda = (4 / 3) * b * π * cs_safe_abs(a) + (b * π * (4 * a + 3 * π * R) *
                                               np.sign(a) / 3)
    dVdb = π * (4 * a + 3 * π * R) * cs_safe_abs(a) / 3
    dVdR = b * π**2 * cs_safe_abs(a)
    return {"a": dVda, "b": dVdb, "R": dVdR}


# not used
def cross_section_area_RZ(R, Z, nx=1000):
    """Cross-sectional area of a poloidal slice

    Parameters
    ----------
    R : fltarray
        radial positions of separatrix
    Z : fltarray
        vertical positions of separatrix
    nx : init
        number of horizontal slices used for integration

    Returns
    -------
    A : float
        cross-sectional area
    """

    x = np.linspace(min(R), max(R), nx)

    pos_inds = (Z >= 0)
    neg_inds = (Z <= 0)

    ypos = scipy.interpolate.interp1d(R[pos_inds],
                                      Z[pos_inds],
                                      bounds_error=False,
                                      fill_value=0)
    yneg = scipy.interpolate.interp1d(R[neg_inds],
                                      Z[neg_inds],
                                      bounds_error=False,
                                      fill_value=0)

    A = np.trapz(ypos(x) - yneg(x), x=x)

    return A


# not used
def surface_area_RZ(R, Z, nx=1000):
    """Surface area of plasma

    Parameters
    ----------
    R : fltarray
        radial positions of separatrix
    Z : fltarray
        vertical positions of separatrix
    nx : init
        number of horizontal slices used for integration

    Returns
    -------
    V : float
        toroidal plasma volume
    """

    x = np.linspace(min(R), max(R), nx)

    pos_inds = (Z >= 0)
    neg_inds = (Z <= 0)

    ypos = scipy.interpolate.interp1d(R[pos_inds],
                                      Z[pos_inds],
                                      bounds_error=False,
                                      fill_value=0)
    yneg = scipy.interpolate.interp1d(R[neg_inds],
                                      Z[neg_inds],
                                      bounds_error=False,
                                      fill_value=0)

    gradypos = np.gradient(ypos(x)) / np.gradient(x)
    S1 = 2 * np.pi * np.trapz(x * np.sqrt(1 + gradypos**2), x=x)

    gradyneg = np.gradient(yneg(x)) / np.gradient(x)
    S2 = 2 * np.pi * np.trapz(x * np.sqrt(1 + gradyneg**2), x=x)

    return S1 + S2


# not used
def volume_RZ(R, Z, nx=1000):
    """Volume of plasma by cylindrical shells

    Parameters
    ----------
    R : fltarray
        radial positions of separatrix
    Z : fltarray
        vertical positions of separatrix
    nx : init
        number of horizontal slices used for integration

    Returns
    -------
    V : float
        toroidal plasma volume
    """

    x = np.linspace(min(R), max(R), nx)

    pos_inds = (Z >= 0)
    neg_inds = (Z <= 0)

    ypos = scipy.interpolate.interp1d(R[pos_inds],
                                      Z[pos_inds],
                                      bounds_error=False,
                                      fill_value=0)
    yneg = scipy.interpolate.interp1d(R[neg_inds],
                                      Z[neg_inds],
                                      bounds_error=False,
                                      fill_value=0)

    # volume by cylindrical shells
    V = 2 * np.pi * np.trapz(x * (ypos(x) - yneg(x)), x=x)

    return V
