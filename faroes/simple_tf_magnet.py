import openmdao.api as om
import numpy as np

from scipy.constants import mu_0, mega

from faroes.configurator import UserConfigurator, Accessor
from importlib import resources


class WindingPackProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        acc = Accessor(self.options['config'])
        f = acc.accessor(["magnet_geometry", "winding pack"])
        acc.set_output(self, f, "f_HTS")

        f = acc.accessor(["materials", "winding pack"])
        acc.set_output(self, f, "j_eff_max", units='MA/m**2')
        acc.set_output(self, f, "B_max", units='T')

        if self.options['config'] is not None:
            f = acc.accessor(["materials", "HTS cable"])
            max_strain = f(["strain limit"])
            youngs = f(["Young's modulus"], units="GPa")
            self.add_output("Young's modulus", youngs, units="GPa")
            self.add_output("max_strain", max_strain)
            self.add_output("max_stress", max_strain * youngs, units="GPa")

        else:
            self.add_output("Young's modulus", units="GPa")
            self.add_output("max_strain")
            self.add_output("max_stress", units="MPa")


class MagnetStructureProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        acc = Accessor(self.options['config'])
        f = acc.accessor(["materials", "structural steel"])
        acc.set_output(self, f, "Young's modulus", units="GPa")


class InnerTFCoilTension(om.ExplicitComponent):
    r"""Total vertical tension on the inner leg of the TF coil


    .. math::

       k & \equiv \log(r2/r1)\\
       T_1 &= \frac{1}{2} I_\mathrm{leg} B_0 R_0 \frac{r_1 + r_2 (k -1)}{r2-r1}

    Assumes a thin current-carrying TF coil loop.
    This can be easily derived by finding the total vertical force on the
    two magnet legs, which depends only on :math:`r1, r2, I_\mathrm{leg}`.
    The linear force density varies inversely with radius. The factor
    :math:`1/2` comes about by assuming that the field decreases linearly
    across the conductor. The tensions in the inner and outer legs
    can be found by ensuring the torque around the 'center of force' is zero.
    The inner leg tension is larger than the outer one, and is more limiting
    since the inner build is more space-constrained.

    Inputs
    ------
    R0 : float
         m, major radius
    B0 : float
         T, field on axis
    r1 : float
         m, radius of inner leg's current carriers
    r2 : float
         m, radius of outer leg's current carriers
    I_leg : float
         MA, current in the coil

    Outputs
    -------
    T1 : float
        N, tension in the inner coil leg

    """
    def setup(self):
        self.add_input('I_leg', units='MA', desc='Current in one leg')
        self.add_input('B0', units='T', desc='Field on axis')
        self.add_input('R0', units='m', desc='Major radius')
        self.add_input('r1',
                       units='m',
                       desc='avg radius of conductor inner leg, at midplane')
        self.add_input('r2',
                       units='m',
                       desc='avg radius of conductor outer leg, at midplane')
        self.add_output('T1', units='MN', desc='Tension on the inner leg')

    def setup_partials(self):
        self.declare_partials('T1', ['I_leg', 'B0', 'R0', 'r1', 'r2'])

    def compute(self, inputs, outputs):
        i_leg = inputs['I_leg']
        b0 = inputs['B0']
        R0 = inputs['R0']
        r1 = inputs['r1']
        r2 = inputs['r2']

        k = np.log(r2 / r1)
        T1 = 0.5 * i_leg * b0 * R0 * (r1 + r2 * (k - 1)) / (r2 - r1)
        outputs['T1'] = T1

    def compute_partials(self, inputs, J):
        """ Jacobian of partial derivatives """

        i_leg = inputs['I_leg']
        b0 = inputs['B0']
        R0 = inputs['R0']
        r1 = inputs['r1']
        r2 = inputs['r2']

        k = np.log(r2 / r1)
        J['T1', 'I_leg'] = b0 * R0 * (r1 + r2 * (k - 1)) / ((r2 - r1) * 2)
        J['T1', 'R0'] = i_leg * b0 * (r1 + r2 * (k - 1)) / ((r2 - r1) * 2)
        J['T1', 'B0'] = i_leg * R0 * (r1 + r2 * (k - 1)) / ((r2 - r1) * 2)
        J['T1', 'r1'] = i_leg * R0 * b0 * r2 * (r1 * (k + 1) - r2) \
            / (2 * r1 * (r1 - r2)**2)
        J['T1', 'r2'] = - i_leg * R0 * b0 * (r1 * (k + 1) - r2) \
            / (2 * (r1 - r2)**2)


class FieldAtRadius(om.ExplicitComponent):
    r"""Toroidal field at R0 and on the inner TF magnet leg

    .. math::

        B_0 = \mu_0 I_\mathrm{tot} / (2 \pi R_0)

        B_\mathrm{on-coil} = \mu_0 I_\mathrm{tot} / (2 \pi r_\mathrm{om})

    Assumes circular symmetry; no ripple

    Inputs
    ------
    I_leg : float
        MA, Current in one TF leg
    r_om : float
        m, Radius of the outermost conductor
    R0 : float
        m, major radius
    n_coil: int
        number of TF coils

    Outputs
    -------
    B0 : float
        T, central field
    B_on_coil : float
        T, maximum toroidal field on the TF coil conductor
    """
    def setup(self):
        self.add_input('I_leg', units='MA')
        self.add_input('r_om', units='m')
        self.add_input('R0', units='m')
        self.add_discrete_input('n_coil', 18)

        self.add_output('B_on_coil', units='T')
        self.add_output('B0', units='T')

    def field_at_radius(self, i, r):
        """Toroidal field at a given radius

        Parameters
        ----------
        i : float
            total current, A
        r : float
            radius, meters

        Returns
        -------
        B : float
            field in Tesla
        """
        b = mu_0 * i / (2 * np.pi * r)
        return b

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        n_coil = discrete_inputs['n_coil']
        I_leg_MA = inputs['I_leg']
        R_coil_max = inputs['r_om']
        R0 = inputs['R0']

        i_total_A = mega * n_coil * I_leg_MA

        B_on_coil = self.field_at_radius(i_total_A, R_coil_max)
        B0 = self.field_at_radius(i_total_A, R0)

        outputs['B_on_coil'] = B_on_coil
        outputs['B0'] = B0

    def setup_partials(self):
        self.declare_partials('B0', ['I_leg', 'R0'])
        self.declare_partials('B_on_coil', ['I_leg', 'r_om'])

    def compute_partials(self, inputs, J, discrete_inputs):
        n_coil = discrete_inputs['n_coil']
        I_leg_MA = inputs['I_leg']
        R_coil_max = inputs['r_om']
        R0 = inputs['R0']

        # need to be careful about the scaling here, since I_leg is in MA
        J['B0', 'I_leg'] = mu_0 * mega * n_coil / (2 * np.pi * R0)
        J['B0', 'R0'] = - mu_0 * mega * n_coil * I_leg_MA \
            / (2 * np.pi * R0**2)
        J['B_on_coil', 'I_leg'] = mu_0 * mega * n_coil \
            / (2 * np.pi * R_coil_max)
        J['B_on_coil', 'r_om'] = - mu_0 * mega * n_coil * I_leg_MA \
            / (2 * np.pi * R_coil_max**2)


class InnerTFCoilStrain(om.ExplicitComponent):
    r"""Strain on the TF coil HTS material

    Assumes that the structure and winding pack strain together.

    .. math::

        T_1 = \sigma_\mathrm{HTS} \left((A_m + A_t)
           \frac{\sigma_\mathrm{struct}}{\sigma_\mathrm{HTS}}
           + f_\mathrm{HTS} A_m\right)

    Inputs
    ------
    T1 : float
        MN, tension on the inner TF coil leg
    A_s : float
        m^2, inner TF leg inner structural area
    A_t : float
        m^2, inner TF leg outer structural area
    A_m : float
        m^2, inner TF leg winding pack area
    f_HTS : float
        fraction of winding pack area which is the superconducting cable

    Outputs
    -------
    s_HTS : float
        MPa, strain on the HTS material
    constraint_max_stress: float
        fraction of the maximum allowed stress on the HTS material
    """
    def setup(self):
        self.add_input('T1', units='MN', desc='Tension on the inner leg')
        self.add_input('A_s',
                       units='m**2',
                       desc='Inner TF leg inner structure area')
        self.add_input('A_t',
                       units='m**2',
                       desc='Inner TF leg outer structure area')
        self.add_input('A_m',
                       units='m**2',
                       desc='Inner TF leg winding pack area')
        self.add_input('f_HTS',
                       desc='Fraction of magnet area which is SC cable')

        self.add_input('hts_max_stress', units='MPa')
        self.add_input('hts_E_young', units='GPa')
        self.add_input('struct_E_young', units='GPa')

        ref_strain_MPa = 100
        self.add_output('s_HTS',
                        units='MPa',
                        ref=ref_strain_MPa,
                        desc='Strain on the HTS cable')
        self.add_output('constraint_max_stress',
                        desc='fraction of maximum stress on the HTS cable')

    def compute(self, inputs, outputs):
        A_s = inputs['A_s']
        A_m = inputs['A_m']
        A_t = inputs['A_t']
        σ_hts_max = inputs['hts_max_stress']
        f_HTS = inputs['f_HTS']
        T1 = inputs['T1']
        struct_E_y = inputs['struct_E_young']
        hts_E_y = inputs['hts_E_young']
        E_rat = struct_E_y / hts_E_y

        sigma_HTS = T1 / ((A_s + A_t) * E_rat + f_HTS * A_m)
        outputs['s_HTS'] = sigma_HTS
        outputs['constraint_max_stress'] = (σ_hts_max - sigma_HTS) / σ_hts_max

    def setup_partials(self):
        self.declare_partials('s_HTS', [
            'A_s', 'A_m', 'A_t', 'f_HTS', 'T1', 'struct_E_young', 'hts_E_young'
        ])
        self.declare_partials('constraint_max_stress', [
            'A_s', 'A_m', 'A_t', 'f_HTS', 'T1', 'hts_max_stress',
            'struct_E_young', 'hts_E_young'
        ])

    def compute_partials(self, inputs, J):
        A_s = inputs['A_s']
        A_m = inputs['A_m']
        A_t = inputs['A_t']
        f_HTS = inputs['f_HTS']
        T1 = inputs['T1']
        σ_hts_max = inputs['hts_max_stress']

        struct_E_y = inputs['struct_E_young']
        hts_E_y = inputs['hts_E_young']
        E_rat = struct_E_y / hts_E_y

        denom = ((A_s + A_t) * E_rat + f_HTS * A_m)
        J['s_HTS', 'T1'] = 1 / denom
        J['s_HTS', 'A_s'] = -E_rat * T1 / denom**2
        J['s_HTS', 'A_t'] = J['s_HTS', 'A_s']
        J['s_HTS', 'A_m'] = -f_HTS * T1 / denom**2
        J['s_HTS', 'f_HTS'] = -A_m * T1 / denom**2
        J['s_HTS', 'struct_E_young'] = -(A_s + A_t) * T1 / (hts_E_y * denom**2)
        J['s_HTS',
          'hts_E_young'] = (A_s + A_t) * struct_E_y * T1 / (hts_E_y**2 *
                                                            denom**2)

        J['constraint_max_stress', 'T1'] = -1 / (denom * σ_hts_max)
        J['constraint_max_stress', 'A_s'] = -J['s_HTS', 'A_s'] / σ_hts_max
        J['constraint_max_stress', 'A_t'] = -J['s_HTS', 'A_t'] / σ_hts_max
        J['constraint_max_stress', 'A_m'] = -J['s_HTS', 'A_m'] / σ_hts_max
        J['constraint_max_stress', 'f_HTS'] = -J['s_HTS', 'f_HTS'] / σ_hts_max
        J['constraint_max_stress',
          'hts_max_stress'] = T1 / (denom * σ_hts_max**2)
        J['constraint_max_stress',
          'struct_E_young'] = (A_s + A_t) * T1 / (hts_E_y * denom**2 *
                                                  σ_hts_max)
        J['constraint_max_stress',
          'hts_E_young'] = -(A_s + A_t) * struct_E_y * T1 / (
              hts_E_y**2 * denom**2 * σ_hts_max)


class MagnetGeometry(om.ExplicitComponent):
    r"""Footprint of the inner and outer TF coil legs

    This closely follows Menard's spreadsheet model, though it may not
    be exactly identical. The TF coil inboard leg is a trapezoidal wedge shape.
    It's divided into three segments: an inner structure, a winding pack,
    and an outer structure. The divisions between the segments are straight
    lines regions parallel to the inner and outer trapezoid walls. They have
    a width, e_gap.

    The measurements "r" are in the direction of the long sides of the
    trapezoid, from origin to outer vertex. The measurements "w"
    for "width" are along the segment from the origin
    to the center of the outer side. The measurements "l" are in the
    direction perpendicular to the "w" measurements.

    r1 and r2 are important for the magnetic force calculations. They
    represent average locations of the current in the winding packs
    of the inboard and outboard legs, respectively.

    The geometry of the outboard leg is less well-defined in this model.
    It is assumed that the distance from the outermost part of the inboard
    leg outer structure to r1 is the same as the distance from the innermost
    part of the outboard leg inner structure to r2. The circumferential width
    "l" of the outboard leg is not described here. For the outboard leg,
    the distinction between 'r' and 'w' measurements is dropped.

    .. image:: images/magnet_cross_section_map.png

    Inputs
    ------
    r_is : float
        m, Inner radius of the inner structure of the inboard leg.
    r_im : float
        m, Inner radius of the winding pack of the inboard leg.
    r_ot : float
        m, Outer radius of the outer structure of the inboard leg.
    n_coil : int
        m, Number of TF coils
    r_iu : float
        m, Inner radius of the inner structure of the outboard leg.

    Outputs
    -------
    A_s : float
        m^2, Cross-sectional area of the inboard leg inner structure.
    A_m : float
        m^2, Cross-sectional area of the inboard leg winding pack.
    A_t : float
        m^2, Cross-sectional area of the inboard leg outer structure.
    r1 : float
        m, Average radius of the winding pack of the inboard leg.
    r2 : float
        m, Average radius of the winding pack of the outer leg.
    r_ov : float,
        m, Outer radius of the outboard leg.
    approximate cross section : float
        m**2, Approximate magnet cross section; expands the trapezoid to a full
           rectangle

    r_os : float
        m, Outer radius of the inner structure of the inboard leg.
    r_om : float
        m, Outer radius of the winding pack of the inboard leg.
    r_it : float
        m, Inner radius of the outer structure of the inboard leg.
    w_is: float
        m, Inner 'width' of the inner structure of the inboard leg.
    w_os: float
        m, Outer 'width' of the inner structure of the inboard leg.
    w_im: float
        m, Inner 'width' of the winding pack of the inboard leg.
    w_om: float
        m, Outer 'width' of the winding pack of the inboard leg.
    w_it: float
        m, Inner 'width' of the outer structure of the inboard leg.
    w_ot: float
        m, Outer 'width' of the outer structure of the inboard leg.

    l_is: float
        m, Inner 'length' of the inner structure of the inboard leg.
    l_os: float
        m, Outer 'length' of the inner structure of the inboard leg.
    l_im: float
        m, Inner 'length' of the winding pack of the inboard leg.
    l_om: float
        m, Outer 'length' of the winding pack of the inboard leg.
    l_it: float
        m, Inner 'length' of the outer structure of the inboard leg.
    l_ot: float
        m, Outer 'length' of the outer structure of the inboard leg.

    """
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        if self.options['config'] is not None:
            config = self.options['config'].accessor(["magnet_geometry"])

            # e_gap is the space between the inner structure and the
            # winding pack, and between the winding pack and the outer
            # structure.
            ground_wrap = config(["ground wrap thickness"], units="m")
            inter_block = config(["inter-block clearance"], units="m")

            self.e_gap = (ground_wrap + inter_block)

            # radial thickness of the A_t external structure
            self.Δr_t = config(["external structure thickness"], units="m")
        else:
            # defaults for testing
            self.e_gap = 0.006
            self.Δr_t = 0.05

        self.add_input('r_is', units='m')
        self.add_input('r_im', units='m')
        self.add_input('r_ot',
                       units='m',
                       desc='Magnet inboard leg outer structure radius')
        self.add_discrete_input('n_coil', val=18, desc='number of coils')
        self.add_input('r_iu',
                       units='m',
                       desc='Inner radius of outboard TF leg')

        self.add_output('r_it', units='m')
        self.add_output('r_os', units='m')
        self.add_output('r_om', units='m')

        # self.add_output('w_is', units='m')
        # self.add_output('w_os', units='m')
        # self.add_output('w_im', units='m')
        # self.add_output('w_om', units='m')
        # self.add_output('w_it', units='m')
        # self.add_output('w_ot', units='m')

        # self.add_output('l_is', units='m')
        # self.add_output('l_os', units='m')
        # self.add_output('l_im', units='m')
        # self.add_output('l_om', units='m')
        # self.add_output('l_it', units='m')
        # self.add_output('l_ot', units='m')

        self.add_output('A_s', units='m**2')
        self.add_output('A_t', units='m**2')
        self.add_output('A_m', units='m**2')

        self.add_output('r1', units='m')
        self.add_output('r2', units='m')
        self.add_output('r_ov', units='m')

        self.add_output('approximate cross section', units='m**2')

        self.add_output('r_im_is_constraint', units='m')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r_is = inputs['r_is']
        r_im = inputs['r_im']
        r_ot = inputs['r_ot']

        r_iu = inputs['r_iu']

        n_coil = discrete_inputs['n_coil']

        r_it = r_ot - self.Δr_t
        r_om = r_it - self.e_gap
        r_os = r_im - self.e_gap

        outputs['r_it'] = r_it
        outputs['r_os'] = r_os
        outputs['r_om'] = r_om

        r_to_w = np.cos(np.pi / n_coil)
        r_to_l = 2 * np.sin(np.pi / n_coil)

        w_is = r_is * r_to_w
        w_os = r_os * r_to_w
        w_im = r_im * r_to_w
        w_om = r_om * r_to_w
        w_it = r_it * r_to_w
        w_ot = r_ot * r_to_w

        # outputs['w_is'] = w_is
        # outputs['w_os'] = w_os
        # outputs['w_im'] = w_im
        # outputs['w_om'] = w_om
        # outputs['w_it'] = w_it
        # outputs['w_ot'] = w_ot

        l_is = r_is * r_to_l
        l_os = r_os * r_to_l
        l_im = r_im * r_to_l
        l_om = r_om * r_to_l
        l_it = r_it * r_to_l
        l_ot = r_ot * r_to_l

        # outputs['l_is'] = l_is
        # outputs['l_os'] = l_os
        # outputs['l_im'] = l_im
        # outputs['l_om'] = l_om
        # outputs['l_it'] = l_it
        # outputs['l_ot'] = l_ot

        outputs['A_s'] = (w_os - w_is) * (l_os + l_is) / 2
        outputs['A_m'] = (w_om - w_im) * (l_om + l_im) / 2
        outputs['A_t'] = (w_ot - w_it) * (l_ot + l_it) / 2

        outputs['r1'] = (r_om + r_im) / 2
        outputs['r2'] = r_iu + (r_ot - r_is) / 2
        outputs['r_ov'] = r_iu + (r_ot - r_is)

        outputs['approximate cross section'] = (r_ot - r_is) * l_ot

        outputs['r_im_is_constraint'] = r_im - self.e_gap - r_is

    def setup_partials(self):
        self.declare_partials('r_it', ['r_ot'], val=1)
        self.declare_partials('r_os', ['r_im'], val=1)
        self.declare_partials('r_om', ['r_ot'], val=1)

        self.declare_partials('A_m', ['r_ot', 'r_im'])
        self.declare_partials('A_t', ['r_ot'])
        self.declare_partials('A_s', ['r_is', 'r_im'])

        self.declare_partials('r1', ['r_ot', 'r_im'], val=1 / 2)
        self.declare_partials('r2', 'r_ot', val=1 / 2)
        self.declare_partials('r2', 'r_is', val=-1 / 2)
        self.declare_partials('r2', 'r_iu', val=1)
        self.declare_partials('r_ov', 'r_ot', val=1)
        self.declare_partials('r_ov', 'r_is', val=-1)
        self.declare_partials('r_ov', 'r_iu', val=1)
        self.declare_partials('r_im_is_constraint', 'r_im', val=1)
        self.declare_partials('r_im_is_constraint', 'r_is', val=-1)

        self.declare_partials('approximate cross section', ['r_ot', 'r_is'])

    def compute_partials(self, inputs, J, discrete_inputs):
        n_coil = discrete_inputs['n_coil']

        whole_ang = np.sin(2 * np.pi / n_coil)
        r_to_l = 2 * np.sin(np.pi / n_coil)

        r_im = inputs['r_im']
        r_is = inputs['r_is']
        r_ot = inputs['r_ot']
        J['A_s', 'r_im'] = (r_im - self.e_gap) * whole_ang
        J['A_s', 'r_is'] = -(r_is) * whole_ang

        J['A_m', 'r_im'] = -r_im * whole_ang
        J['A_m', 'r_ot'] = -(self.e_gap - r_ot + self.Δr_t) * whole_ang
        J['A_t', 'r_ot'] = self.Δr_t * whole_ang

        J['approximate cross section', 'r_ot'] = (2 * r_ot - r_is) * r_to_l
        J['approximate cross section', 'r_is'] = -r_ot * r_to_l


class MagnetCurrent(om.ExplicitComponent):
    r"""Calculate the current per TF coil leg

    .. math::
        I_\mathrm{leg} = A_m f_\mathrm{HTS} j_\mathrm{HTS}

    Inputs
    ------
    A_m : float
        m^2, area of the inboard leg winding pack
    f_HTS : float
        Fraction of that winding pack which is superconducting cable
    j_HTS : float
        MA/m^2, Current density in that superconducting cable

    Outputs
    -------
    I_leg : float
        MA: Current in one TF leg
    j_eff_wp_max : float
        MA/m^2 : maximum current density in the superconducting cable

    """
    def setup(self):
        self.add_input('A_m', units='m**2')
        self.add_input('f_HTS')
        self.add_input('j_HTS', units='MA/m**2')
        self.add_output('I_leg', units='MA')

    def compute(self, inputs, outputs):
        A_m = inputs['A_m']
        f_HTS = inputs['f_HTS']
        j_HTS = inputs['j_HTS']
        i_leg = A_m * f_HTS * j_HTS
        outputs['I_leg'] = i_leg

    def setup_partials(self):
        self.declare_partials('I_leg', ['A_m', 'f_HTS', 'j_HTS'])

    def compute_partials(self, inputs, J):
        A_m = inputs['A_m']
        f_HTS = inputs['f_HTS']
        j_HTS = inputs['j_HTS']
        J['I_leg', 'A_m'] = f_HTS * j_HTS
        J['I_leg', 'f_HTS'] = A_m * j_HTS
        J['I_leg', 'j_HTS'] = A_m * f_HTS


class MagnetRadialBuild(om.Group):
    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']

        self.add_subsystem(
            'geometry',
            MagnetGeometry(config=config),
            promotes_inputs=['r_ot', 'n_coil', 'r_iu', 'r_im', 'r_is'],
            promotes_outputs=[
                'A_s', 'A_t', 'A_m', 'r1', 'r2', 'r_om', 'r_im_is_constraint',
                ('r_ov', 'Ob TF R_out'), 'approximate cross section'
            ])
        self.add_subsystem('windingpack',
                           WindingPackProperties(config=config),
                           promotes=['f_HTS', 'B_max'])
        self.add_subsystem('magnetstructure_props',
                           MagnetStructureProperties(config=config))
        self.add_subsystem('current',
                           MagnetCurrent(),
                           promotes_inputs=['A_m', 'f_HTS', 'j_HTS'],
                           promotes_outputs=['I_leg'])
        self.add_subsystem('field',
                           FieldAtRadius(),
                           promotes_inputs=['I_leg', 'r_om', 'R0', 'n_coil'],
                           promotes_outputs=['B_on_coil', 'B0'])
        self.add_subsystem('tension',
                           InnerTFCoilTension(),
                           promotes_inputs=['I_leg', 'r1', 'r2', 'R0', 'B0'],
                           promotes_outputs=['T1'])
        self.add_subsystem(
            'strain',
            InnerTFCoilStrain(),
            promotes_inputs=['T1', 'A_m', 'A_t', 'A_s', 'f_HTS'],
            promotes_outputs=['s_HTS', 'constraint_max_stress'])

        self.add_subsystem('obj_cmp',
                           om.ExecComp('obj = -B0', B0={'units': 'T'}),
                           promotes=['B0', 'obj'])
        self.add_subsystem(
            'con_cmp2',
            om.ExecComp('constraint_B_on_coil = B_max - B_on_coil',
                        B_on_coil={'units': 'T'},
                        B_max={'units': 'T'}),
            promotes=['constraint_B_on_coil', 'B_on_coil', 'B_max'])
        self.add_subsystem(
            'con_cmp3',
            om.ExecComp(
                'constraint_wp_current_density = A_m * j_eff_wp_max - I_leg',
                A_m={'units': 'm**2'},
                j_eff_wp_max={'units': 'MA/m**2'},
                I_leg={'units': 'MA'}),
            promotes=['constraint_wp_current_density', 'A_m', 'I_leg'])

        self.connect('windingpack.max_stress', ['strain.hts_max_stress'])
        self.connect("windingpack.Young's modulus", ['strain.hts_E_young'])
        self.connect("magnetstructure_props.Young's modulus",
                     ['strain.struct_E_young'])
        self.connect('windingpack.j_eff_max', ['con_cmp3.j_eff_wp_max'])


if __name__ == "__main__":
    prob = om.Problem()

    resource_dir = 'faroes.test.test_data'
    with resources.path(resource_dir,
                        'config_menard_spreadsheet.yaml') as path:
        uc = UserConfigurator(path)

    prob.model = MagnetRadialBuild(config=uc)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    prob.model.add_design_var('r_is', lower=0.03, upper=0.4)
    prob.model.add_design_var('r_im', lower=0.05, upper=0.5)
    prob.model.add_design_var('j_HTS', lower=0, upper=300)

    prob.model.add_objective('obj')

    # set constraints
    prob.model.add_constraint('constraint_max_stress', lower=0)
    prob.model.add_constraint('constraint_B_on_coil', lower=0)
    prob.model.add_constraint('constraint_wp_current_density', lower=0)
    prob.model.add_constraint('A_s', lower=0)

    prob.setup()

    prob.set_val('R0', 3, 'm')
    prob.set_val('n_coil', 18)
    prob.set_val('geometry.r_ot', 0.405, 'm')
    prob.set_val('geometry.r_iu', 8.025, 'm')
    prob.set_val('windingpack.max_stress', 525, "MPa")
    prob.set_val('windingpack.max_strain', 0.003)
    prob.set_val("windingpack.Young's modulus", 175, "GPa")
    prob.set_val('windingpack.j_eff_max', 160, "MA/m**2")
    prob.set_val('windingpack.f_HTS', 0.76)
    #    prob.set_val("magnetstructure_props.Young's modulus", 220)

    prob.run_driver()

    # prob.model.list_inputs(values=True)
    prob.model.list_outputs(values=True)
