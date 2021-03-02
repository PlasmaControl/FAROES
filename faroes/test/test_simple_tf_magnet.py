import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials
import unittest

import numpy.random as nprand

import faroes.simple_tf_magnet as magnet


class TestSimpleTFMagnet(unittest.TestCase):
    def test_magnet_build(self):
        prob = om.Problem()

        prob.model = magnet.MagnetRadialBuild()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = False

        prob.model.add_design_var('r_is', lower=0.03, upper=0.4, units="m")
        prob.model.add_design_var('f_im', lower=0.05, upper=0.95)
        prob.model.add_design_var('j_HTS', lower=0, upper=300, units="MA/m**2")

        prob.model.add_objective('B0', scaler=-1)

        # set constraints
        prob.model.add_constraint('constraint_max_stress', lower=0)
        prob.model.add_constraint('constraint_B_on_coil', lower=0)
        prob.model.add_constraint('constraint_wp_current_density', lower=0)
        prob.model.add_constraint('A_s', lower=0)

        prob.setup()

        prob.set_val('R0', 3)
        prob.set_val('n_coil', 18)
        prob.set_val('r_is', 0.1)
        prob.set_val('geometry.r_ot', 0.405)
        prob.set_val('geometry.r_iu', 8.025)

        prob.set_val('windingpack.max_stress', 525, units="MPa")
        prob.set_val('windingpack.j_eff_max', 160)
        prob.set_val('windingpack.f_HTS', 0.76)
        prob.set_val('windingpack.B_max', 18, units="T")
        prob.set_val("magnetstructure_props.Young's modulus", 220)

        prob.run_driver()

        assert_near_equal(prob['B0'][0], 2.094, 1e-3)
        assert_near_equal(prob['B_on_coil'][0], 18, 1e-3)

        # check radius order
        self.assertTrue(0 < prob['r_is'][0])
        self.assertTrue(prob['geometry.r_is'][0] < prob['geometry.r_os'][0])
        self.assertTrue(prob['geometry.r_os'][0] < prob['geometry.r_im'][0])
        self.assertTrue(prob['geometry.r_im'][0] < prob['geometry.r1'][0])
        self.assertTrue(prob['geometry.r1'][0] < prob['geometry.r_om'][0])
        self.assertTrue(prob['geometry.r_om'][0] < prob['geometry.r_it'][0])
        self.assertTrue(prob['geometry.r_im'][0] < prob['geometry.r_ot'][0])
        self.assertTrue(prob['geometry.r_iu'][0] < prob['geometry.r2'][0])

        self.assertTrue(prob['T1'][0] > 0)
        self.assertTrue(prob['I_leg'][0] > 0)


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

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMagnetGeometry(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = magnet.MagnetGeometry()
        prob.setup(force_alloc_complex=True)
        prob['r_is'] = 0.5 * nprand.random()
        prob['f_im'] = nprand.random()
        prob['r_iu'] = 8 + nprand.random()
        prob['r_ot'] = 1.5 + nprand.random()
        prob['n_coil'] = 18
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

        prob['A_m'] = nprand.random()
        prob['f_HTS'] = nprand.random()
        prob['j_HTS'] = nprand.random()

        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check)


class TestInnerTFCoilStrain(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = magnet.InnerTFCoilStrain()
        prob.setup(force_alloc_complex=True)

        prob['T1'] = nprand.random()
        prob['A_s'] = nprand.random() + 0.01
        prob['A_m'] = nprand.random() + 0.01
        prob['A_t'] = nprand.random() + 0.01
        prob['f_HTS'] = nprand.random()

        check = prob.check_partials(out_stream=None,
                                    compact_print=True,
                                    method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
