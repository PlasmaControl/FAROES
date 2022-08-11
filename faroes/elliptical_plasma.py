from faroes.configurator import UserConfigurator
from faroes.shapinglimits import MenardKappaScaling
import faroes.util as util

import openmdao.api as om
from math import pi as π


class EllipseLikeGeometry(om.ExplicitComponent):
    r"""Describes an elliptical or ellipse-like plasma.

    .. image :: images/ellipticalplasmageometry.png
       :width: 300
       :alt: A vertically-oriented ellipse with horizontal semi-minor axis 'a',
           vertical semi-major axis 'b'. The center of the ellipse is at a
           radius 'R0' from the central vertical axis of the torus.

    .. math::
        a &= R_0 / A

        δ &= 0

    :math:`κ_a` is an 'effective elongation' which is the same
    as :math:`κ` for a perfect ellipse plasma, but can be smaller
    to model a more realistic diverted plasma shape or volume.
    The "ellipse-like plasma" has :math:`κ_a \neq κ`.

    Other properties are determined using standard geometry formulas:

    .. math::
        b &= κ a \\
        ε &= 1 / A \\
        \textrm{full plasma height} &= 2 b \\
        L_\mathrm{pol} &= \textrm{ellipse_perimeter(a, b)} \\
        \textrm{surface area} &= 2 \pi R \, \textrm{ellipse_perimeter(a, b)} \\
        R_\mathrm{in} &= R - a \\
        R_\mathrm{out} &= R + a \\
        V &= 2 \pi R \, \pi a^2 κ_a \\
        S_c &= \pi a^2 κ_a .

    where :math:`L_\mathrm{pol}` is the poloidal circumference and :math:`S_c`
    is the poloidal cross-section area.

    Note that R0, a, and A must be consistent with each other. The three
    inputs are specified to avoid recomputation and for ease of computation.

    Inputs
    ------
    R0 : float
        m, major radius
    a : float
        m, minor radius
    A : float
        Aspect ratio
    κ : float
        Elongation
    κa : float
        Effective elongation, :math:`S_c / (π a^2)`

        Here :math:`S_c` is the plasma cross-sectional area.
        Same as :math:`κ` for this elliptical plasma.

    Outputs
    -------
    b : float
        m, Minor radius in vertical direction
    δ : float
        Triangularity: is always zero
    ε : float
        Inverse aspect ratio
    full plasma height : float
        m, Twice b
    surface area : float
        m**2, Surface area
    L_pol : float
        m, Poloidal circumference
    L_pol_simple : float
        m, Simplified poloidal circumference for testing.
        Uses a simple ellipse approximation.
    S_c : float
        m**2, Poloidal cross section area
    V : float
        m**3, Plasma volume
    R_in : float
        m, Inner major radius
    R_out : float
        m, Outer major radius
    """
    def setup(self):
        self.add_input("R0", units='m', desc="Major radius")
        self.add_input("A", desc="Aspect Ratio")
        self.add_input("a", units='m', desc="Minor radius")
        self.add_input("κ", desc="Elongation")
        self.add_input("κa", desc="Effective elongation")

        self.add_output("b", units='m', desc="Minor radius height")
        self.add_output("ε", desc="Inverse aspect ratio")
        self.add_output("δ", desc="Triangularity")
        self.add_output("full_plasma_height",
                        units='m',
                        desc="Top to bottom of the ellipse")
        self.add_output("surface area",
                        units='m**2',
                        lower=0,
                        ref=100,
                        desc="Surface area of the elliptical plasma")
        self.add_output("V",
                        units='m**3',
                        desc="Plasma volume",
                        lower=0,
                        ref=100)
        self.add_output("S_c",
                        units='m**2',
                        desc="Cross-sectional area of the plasma",
                        lower=0,
                        ref=20)
        self.add_output("L_pol",
                        units="m",
                        desc="Poloidal circumference of the plasma",
                        lower=0,
                        ref=10)
        self.add_output("L_pol_simple",
                        units="m",
                        lower=0,
                        ref=10,
                        desc="Simplified poloidal circumference for testing")
        # These may be 'recomputations' for larger models
        # but are useful for using this component in a 'standalone' fashion
        self.add_output("R_in",
                        units='m',
                        desc="Inner radius of plasma at midplane")
        self.add_output("R_out",
                        units='m',
                        desc="Outer radius of plasma at midplane")

    def compute(self, inputs, outputs):
        outputs["δ"] = 0  # ellipse-like plasma approximation
        R0 = inputs["R0"]
        A = inputs["A"]
        a = inputs["a"]
        κ = inputs["κ"]
        κa = inputs["κa"]

        b = κ * a
        L_pol = util.ellipse_perimeter(a[0], b[0])
        L_pol_simple = util.ellipse_perimeter_simple(a[0], b[0])
        sa = 2 * π * R0 * L_pol

        outputs["ε"] = 1 / A
        outputs["b"] = b
        outputs["full_plasma_height"] = 2 * b
        outputs["surface area"] = sa

        outputs["L_pol"] = L_pol
        outputs["L_pol_simple"] = L_pol_simple

        outputs["S_c"] = π * a**2 * κa
        V = 2 * π**2 * κa * a**2 * R0
        outputs["V"] = V

        outputs["R_in"] = R0 - a
        outputs["R_out"] = R0 + a

    def setup_partials(self):
        self.declare_partials("b", ["a", "κ"])
        self.declare_partials("ε", ["A"])
        self.declare_partials("full_plasma_height", ["a", "κ"])
        self.declare_partials("L_pol", ["a", "κ"], method="exact")
        self.declare_partials("L_pol_simple", ["a", "κ"], method="exact")
        self.declare_partials("surface area", ["R0", "a", "κ"], method="exact")

        self.declare_partials("V", ["R0", "a", "κa"], method="exact")
        self.declare_partials("S_c", ["a", "κa"], method="exact")

        self.declare_partials(["R_in", "R_out"], "R0", val=1)
        self.declare_partials("R_in", "a", val=-1)
        self.declare_partials("R_out", "a", val=1)

    def compute_partials(self, inputs, J):
        A = inputs["A"]
        a = inputs["a"]
        R0 = inputs["R0"]
        κ = inputs["κ"]
        κa = inputs["κa"]
        b = κ * a
        L_pol = util.ellipse_perimeter(a[0], b[0])

        J["b", "a"] = κ
        J["b", "κ"] = a
        J["full_plasma_height", "a"] = 2 * κ
        J["full_plasma_height", "κ"] = 2 * a

        J["ε", "A"] = -1 / A**2

        dL_pol = util.ellipse_perimeter_derivatives(a[0], b[0])
        dL_pol_da = dL_pol["a"]
        dL_pol_db = dL_pol["b"]
        J["L_pol", "a"] = dL_pol_da + dL_pol_db * J["b", "a"]
        J["L_pol", "κ"] = dL_pol_db * J["b", "κ"]

        dL_pols = util.ellipse_perimeter_simple_derivatives(a[0], b[0])
        dL_pols_da = dL_pols["a"]
        dL_pols_db = dL_pols["b"]
        J["L_pol_simple", "a"] = dL_pols_da + dL_pols_db * J["b", "a"]
        J["L_pol_simple", "κ"] = dL_pols_db * J["b", "κ"]

        J["surface area", "a"] = 2 * π * R0 * J["L_pol", "a"]
        J["surface area", "R0"] = 2 * π * L_pol
        J["surface area", "κ"] = 2 * π * R0 * J["L_pol", "κ"]

        J["V", "R0"] = 2 * a**2 * π**2 * κa
        J["V", "a"] = 4 * a * π**2 * R0 * κa
        J["V", "κa"] = 2 * a**2 * π**2 * R0

        J["S_c", "a"] = π * 2 * a * κa
        J["S_c", "κa"] = π * a**2


class EllipticalPlasmaGeometry(om.Group):
    r"""Perfect ellipse plasma.

    Fixes :math:`κ_a = κ`.

    Uses :class:`.MenardKappaScaling`
    to determine :math:`κ`.

    Otherwise behaves like
    :class:`.EllipseLikeGeometry`.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("kappa",
                           MenardKappaScaling(config=config),
                           promotes_inputs=["A"],
                           promotes_outputs=["κ"])
        # enforce equality
        self.add_subsystem("passthrough",
                           om.ExecComp("kappaa = kappa",
                                       kappa={'desc': "Elongation"},
                                       kappaa={'desc': "Elongation"}),
                           promotes_inputs=[("kappa", "κ")],
                           promotes_outputs=[("kappaa", "κa")])

        self.add_subsystem("geom",
                           EllipseLikeGeometry(),
                           promotes_inputs=["R0", "a", "A", "κ", "κa"],
                           promotes_outputs=["*"])


class MenardPlasmaGeometry(om.Group):
    r"""Not-quite-ellipse plasma shape.

    Loads "κ area fraction" from a configuration file
    and has :math:`κ_a \neq κ`.

    Otherwise behaves like
    :class:`.EllipseLikeGeometry`.
    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        config = self.options['config']
        self.add_subsystem("kappa",
                           MenardKappaScaling(config=config),
                           promotes_inputs=["A"],
                           promotes_outputs=["κ", "κa"])
        self.add_subsystem("geom",
                           EllipseLikeGeometry(),
                           promotes_inputs=["R0", "a", "A", "κ", "κa"],
                           promotes_outputs=["*"])


if __name__ == "__main__":
    uc = UserConfigurator()
    prob = om.Problem()

    prob.model = EllipticalPlasmaGeometry(config=uc)

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('A', 1.6)
    prob.set_val('a', 1.875)
    prob.set_val('κ', 2.7)
    prob.set_val('κa', 2.7)

    prob.run_driver()
    prob.model.list_inputs(desc=True, units=True)
    prob.model.list_outputs(desc=True, units=True)
