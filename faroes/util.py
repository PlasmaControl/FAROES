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
        self.add_input("X0", units="m", val=0)
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

    References
    ----------
    [1] https://mathworld.wolfram.com/ParallelCurves.html
    [2] https://en.wikipedia.org/wiki/Parallel_curve
    """
    def setup(self):
        self.add_input("x", units="m", shape_by_conn=True)
        self.add_input("y", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("dx_dt", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("dy_dt", units="m", copy_shape="x", shape_by_conn=True)
        self.add_input("s", units="m", desc="offset")

        self.add_output("x_o", units="m", copy_shape="x")
        self.add_output("y_o", units="m", copy_shape="x")

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

    def plot(self, ax=None, **kwargs):
        ax.plot(self.get_val('x_o'), self.get_val('y_o'), **kwargs)


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
