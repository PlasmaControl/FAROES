import openmdao.api as om

from faroes.configurator import UserConfigurator
from faroes.simple_tf_magnet import MagnetGeometry, WindingPackProperties
from faroes.simple_tf_magnet import MagnetStructureProperties
from faroes.simple_tf_magnet import MagnetCurrent
from faroes.simple_tf_magnet import FieldAtRadius, InnerTFCoilTension
from faroes.simple_tf_magnet import InnerTFCoilStrain
from faroes.elliptical_coil import SimpleEllipticalTFSet, TFSetProperties

from importlib import resources


class TFMagnetSet(om.Group):
    r"""

    Inputs
    ------
    Δr_s : float
        m, Thickness of inboard inner structure. Useful as a design variable.
    Δr_m : float
        m, Thickness of inboard winding pack. Useful as a design variable.
    j_HTS : float
        MA/m**2, current density in the HTS cables. Useful as a design
        variable.

    R0 : float
        m, plasma major radius
    κ : float
        Plasma elongation
    Ib TF R_in : float
        m, Inboard leg inner radius
    Ib TF R_out : float
        m, Inboard leg outer radius
    Ob TF R_in : float
        m, Outboard leg inner radius

    n_coil : int
        Number of coils in the TF set

    Outputs
    -------
    B0 : float
        T, Vacuum field on axis
    Ob TF R_out : float
        m, Outboard leg outer radius
    I_leg : float
        MA, current per TF leg

    arc length : float
        m, Arc length of each TF magnet
    half-height : float
        m, Half-height of the TF magnets
    V_magnet_structure : float
        m**3, Total volume of the magnet structures
    V_enc : float
        m**3, Volume enclosed by the magnets

    constraint_max_stress : float
        Fraction of maximum vertical stress reached.
    constraint_B_on_coil : float
        Fraction of maximum B on coil
    constraint_wp_current_density : float
        Fraction of maximum current density

    Notes
    -----
    (Additional inputs are specified via the configuration files)
    """

    def initialize(self):
        self.options.declare('config', default=None)

    def setup(self):
        config = self.options['config']

        self.add_subsystem(
            'geometry',
            MagnetGeometry(config=config),
            promotes_inputs=['r_is', 'Δr_s', 'Δr_m',
                             'r_iu', 'n_coil'],
            promotes_outputs=[
                ('r_ot', 'Ib TF R_out'),
                ('r_ov', 'Ob TF R_out'),
            ])

        self.add_subsystem('windingpack',
                           WindingPackProperties(config=config),
                           promotes=['f_HTS', 'B_max'])
        self.add_subsystem('magnetstructure_props',
                           MagnetStructureProperties(config=config))
        self.add_subsystem('current',
                           MagnetCurrent(),
                           promotes_inputs=['f_HTS', 'j_HTS'],
                           promotes_outputs=['I_leg'])
        self.add_subsystem('field',
                           FieldAtRadius(),
                           promotes_inputs=['I_leg', 'R0', 'n_coil'],
                           promotes_outputs=['B_on_coil', 'B0'])
        self.add_subsystem('tension',
                           InnerTFCoilTension(),
                           promotes_inputs=['I_leg', 'R0', 'B0'],
                           promotes_outputs=['T1'])
        self.add_subsystem('strain',
                           InnerTFCoilStrain(),
                           promotes_inputs=['T1', 'f_HTS'],
                           promotes_outputs=['s_HTS', 'constraint_max_stress'])

        self.add_subsystem('profile_props', TFSetProperties(config=config))

        self.add_subsystem('profile',
                           SimpleEllipticalTFSet(config=config),
                           promotes_inputs=["R0", "n_coil", "κ"],
                           promotes_outputs=[
                               "arc length", "half-height",
                               ("V_set", "V_magnet_structure"), "V_enc"
                           ])

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
            promotes=['constraint_wp_current_density', 'I_leg'])

        self.connect('windingpack.max_stress', ['strain.hts_max_stress'])
        self.connect("windingpack.Young's modulus", ['strain.hts_E_young'])
        self.connect("magnetstructure_props.Young's modulus",
                     ['strain.struct_E_young'])
        self.connect('windingpack.j_eff_max', ['con_cmp3.j_eff_wp_max'])

        self.connect('profile_props.elongation_multiplier',
                     ['profile.elongation_multiplier'])
        self.connect('geometry.r_om', ['field.r_om'])
        self.connect('geometry.r1', ['tension.r1', 'profile.r1'])
        self.connect('geometry.r2', ['tension.r2', 'profile.r2'])
        self.connect('geometry.A_m',
                     ['strain.A_m', 'current.A_m', 'con_cmp3.A_m'])
        self.connect('geometry.A_t', ['strain.A_t'])
        self.connect('geometry.A_s', ['strain.A_s'])
        self.connect('geometry.approximate cross section',
                     ['profile.cross section'])


if __name__ == "__main__":
    prob = om.Problem()

    resource_dir = 'faroes.test.test_data'
    with resources.path(resource_dir,
                        'config_menard_spreadsheet.yaml') as path:
        uc = UserConfigurator(path)

    prob.model = TFMagnetSet(config=uc)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    prob.model.add_design_var('r_is', lower=0.03, upper=0.4)
    prob.model.add_design_var('Δr_m', lower=0.05, upper=0.5)
    prob.model.add_design_var('j_HTS', lower=0, upper=300)

    prob.model.add_objective('obj')

    # set constraints
    prob.model.add_constraint('constraint_max_stress', lower=0)
    prob.model.add_constraint('constraint_B_on_coil', lower=0)
    prob.model.add_constraint('constraint_wp_current_density', lower=0)

    prob.setup()
    # prob.check_config(checks=['unconnected_inputs'])

    # initial values for design variables
    prob.set_val('r_is', 0.1, 'm')
    prob.set_val('Δr_m', 0.1, 'm')
    prob.set_val('j_HTS', 10, 'MA/m**2')

    prob.set_val('R0', 3, 'm')
    prob.set_val('Δr_s', 0.1, 'm')
    prob.set_val('κ', 2.7)
    prob.set_val('n_coil', 18)
    prob.set_val('Ib TF R_out', 0.405, 'm')
    prob.set_val('r_iu', 8.025, 'm')
    prob.set_val('windingpack.max_stress', 525, "MPa")
    prob.set_val('windingpack.max_strain', 0.003)
    prob.set_val("windingpack.Young's modulus", 175, "GPa")
    prob.set_val('windingpack.j_eff_max', 160, "MA/m**2")
    prob.set_val('windingpack.f_HTS', 0.76)
    prob.set_val("magnetstructure_props.Young's modulus", 220)

    prob.run_driver()

    # prob.model.list_inputs(values=True)
    # prob.model.list_outputs(values=True)
