import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials
import unittest

import faroes.simple_tf_magnet as magnet

example_value_1 = 0.12345
example_value_2 = 0.54321
example_value_3 = 0.14253


class TestSimpleTFMagnet(unittest.TestCase):
    def test_magnet_build(self):
        prob = om.Problem()

        prob.model = magnet.ExampleMagnetRadialBuild()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = False

        prob.model.add_design_var('r_is', lower=0.10, upper=0.4, units="m")
        prob.model.add_design_var('Δr_s', lower=0.01, upper=0.95, units="m")
        prob.model.add_design_var('Δr_m', lower=0.01, upper=0.95, units="m")
        prob.model.add_design_var('eng.j_HTS',
                                  lower=0,
                                  upper=250,
                                  units="MA/m**2")
        prob.model.add_design_var('dR', lower=0, upper=1, units="m")

        prob.model.add_objective('B0', scaler=-1)

        # set constraints
        prob.model.add_constraint('eng.constraint_max_stress', lower=0)
        prob.model.add_constraint('eng.constraint_B_on_coil', lower=0)
        prob.model.add_constraint('eng.constraint_wp_current_density', lower=0)
        prob.model.add_constraint('Ib TF R_out', equals=0.405, units="m")
        prob.model.add_constraint('A_s', lower=0, units='m**2')

        prob.setup()

        prob.set_val('R0', 3, units='m')
        prob.set_val('r_is', 0.2, units='m')
        prob.set_val('Δr_s', 0.1, units='m')
        prob.set_val('Δr_m', 0.1, units='m')
        prob.set_val('n_coil', 18)
        prob.set_val('ob_gap.r_min', 8.025, 'm')
        prob.set_val('dR', 1.000, 'm')

        prob.set_val('eng.windingpack.max_stress', 525, units='MPa')
        prob.set_val("eng.windingpack.Young's modulus", 175, units='GPa')
        prob.set_val('eng.windingpack.max_strain', 0.003)
        prob.set_val('eng.windingpack.j_eff_max', 160, units='MA/m**2')
        prob.set_val('eng.windingpack.f_HTS', 0.76)
        prob.set_val('eng.windingpack.B_max', 18, units='T')
        prob.set_val("eng.magnetstructure_props.Young's modulus",
                     220,
                     units='GPa')

        prob.run_driver()

        assert_near_equal(prob['B0'][0], 2.094, 1e-3)
        assert_near_equal(prob['eng.B_on_coil'][0], 18, 1e-3)

        # check radius order
        self.assertTrue(0 < prob['r_is'][0])
        self.assertTrue(prob['geometry.r_is'][0] < prob['geometry.r_os'][0])
        self.assertTrue(prob['geometry.r_os'][0] < prob['geometry.r_im'][0])
        self.assertTrue(prob['geometry.r_im'][0] < prob['geometry.r1'][0])
        self.assertTrue(prob['geometry.r1'][0] < prob['geometry.r_om'][0])
        self.assertTrue(prob['geometry.r_om'][0] < prob['geometry.r_it'][0])
        self.assertTrue(prob['geometry.r_im'][0] < prob['geometry.r_ot'][0])
        self.assertTrue(
            prob['ob_geometry.r_iu'][0] < prob['ob_geometry.r2'][0])

        self.assertTrue(prob['eng.tension.T1'][0] > 0)
        self.assertTrue(prob['eng.I_leg'][0] > 0)


class TestFieldAtRadius(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = magnet.FieldAtRadius()
        prob.setup(force_alloc_complex=True)

        prob['n_coil'] = 1
        prob['I_leg'] = 10
        prob['R0'] = 2
        prob['r_om'] = 1
        prob.run_model()
        assert_near_equal(prob['B0'][0], 1, 1e-4)
        assert_near_equal(prob['B_on_coil'][0], 2, 1e-4)

        # Here fd is used to check because the derivatives wrt n_coil are cs.
        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, rtol=2e-6, atol=3e-6)


class TestInboardMagnetGeometry(unittest.TestCase):

    def setUp(self):
        prob = om.Problem()

        prob.model = magnet.InboardMagnetGeometry()
        prob.setup(force_alloc_complex=True)
        prob['r_is'] = 0.5 * example_value_1
        prob['Δr_m'] = example_value_3
        prob['Δr_s'] = example_value_2
        prob['n_coil'] = 18
        prob.run_driver()
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        # once converted to exact derivatives, these could be checked via cs.
        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, rtol=2e-6)


class TestOutboardMagnetGeometry(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = magnet.OutboardMagnetGeometry()
        prob.setup(force_alloc_complex=True)
        prob['Ib TF Δr'] = 1 + 0.5 * example_value_1
        prob['r_iu'] = 4 + 0.5 * example_value_2
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMagnetCurrent(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = magnet.MagnetCurrent()
        prob.setup()

        prob['A_m'] = example_value_1
        prob['f_HTS'] = example_value_2
        prob['j_HTS'] = example_value_3
        prob.run_driver()

        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check)


class TestInnerTFCoilStrain(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = magnet.InnerTFCoilStrain()
        prob.setup(force_alloc_complex=True)

        prob['T1'] = example_value_1
        prob['A_s'] = example_value_2 + 0.01
        prob['A_m'] = example_value_3 + 0.01
        prob['A_t'] = example_value_1 + 0.01
        prob['f_HTS'] = example_value_2

        check = prob.check_partials(out_stream=None,
                                    compact_print=True,
                                    method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
