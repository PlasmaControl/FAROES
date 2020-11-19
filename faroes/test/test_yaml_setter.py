import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import faroes.yaml_setter
import unittest


class TestYamlValidateLeaf(unittest.TestCase):
    def setUp(self):
        self.f = faroes.yaml_setter.validateleaf

    def test_num(self):
        self.assertIsNone(self.f(3))

    def test_leaf_good(self):
        self.assertIsNone(self.f({'value': 405, 'units': 'mm'}))

    def test_leaf_good_two(self):
        self.assertIsNone(self.f({'value': 405}))

    def test_leaf_bad(self):
        self.assertRaisesRegex(KeyError, "Unit specified without value",
                               self.f, {'units': 'm'})

    def test_leaf_bad_two(self):
        self.assertRaisesRegex(KeyError, "No value", self.f, {'Reference': 3})


class TestYamlIsLeaf(unittest.TestCase):
    def setUp(self):
        self.f = faroes.yaml_setter.is_leaf

    def test_num(self):
        self.assertTrue(self.f(3))

    def test_dict(self):
        self.assertTrue(self.f({'value': 3}))

    def test_dict_array(self):
        self.assertTrue(self.f({'value': [1, 2, 3], 'units': 'm'}))

    def test_dict_subdict(self):
        self.assertTrue(self.f({'value': {'a': 1, 'b': 2}, 'units': 'm'}))

    def test_dict_reference(self):
        self.assertTrue(
            self.f({
                'value': {
                    'a': 1,
                    'b': 2
                },
                'units': 'm',
                'reference': 'asdf'
            }))

    def test_not_dict_subdict(self):
        self.assertFalse(self.f({'a': {'value': 1}, 'reference': 'asdf'}))


class TestFullProblem(unittest.TestCase):
    def test_full_loading(self):
        from faroes.simple_tf_magnet import MagnetRadialBuild

        yaml_here_string = """
        R0 : 3
        n_coil : 18
        geometry:
            r_ot: {'value': 405, 'units': 'mm'}
            r_iu: 8.025
        windingpack:
            f_HTS: 0.76
            j_eff_max : {'value' : 160, 'units': 'MA/m**2'}
        """

        prob = om.Problem()

        prob.model = MagnetRadialBuild()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('r_is', lower=0.03, upper=0.4)
        prob.model.add_design_var('r_im', lower=0.05, upper=0.5)
        prob.model.add_design_var('j_HTS', lower=0, upper=300)

        prob.model.add_objective('obj')

        # set constraints
        prob.model.add_constraint('max_stress_con', lower=0)
        prob.model.add_constraint('constraint_B_on_coil', lower=0)
        prob.model.add_constraint('constraint_wp_current_density', lower=0)

        prob.setup()

        faroes.yaml_setter.load_yaml_to_problem(prob, yaml_here_string)

        prob.run_driver()
        assert_near_equal(prob['B0'][0], 2.094, 1e-3)
        assert_near_equal(prob['B_on_coil'][0], 18, 1e-3)
        assert_near_equal(prob['n_coil'], 18, 1e-3)
        assert_near_equal(prob['geometry.r_ot'][0], 0.405, 1e-3)
        assert_near_equal(prob['windingpack.j_eff_max'][0], 160, 1e-3)


if __name__ == '__main__':
    unittest.main()
