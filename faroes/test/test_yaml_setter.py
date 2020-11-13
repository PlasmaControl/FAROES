import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from faroes.yaml_setter import *
import unittest


class TestYamlProblem(unittest.TestCase):
    def test_validate_leaf_num(self):
        self.assertIsNone(validateleaf(3))

    def test_validate_dict_leaf_good(self):
        self.assertIsNone(validateleaf({'value': 405, 'units': 'mm'}))

    def test_validate_dict_leaf_good_two(self):
        self.assertIsNone(validateleaf({'value': 405}))

    def test_validate_dict_leaf_bad(self):
        self.assertRaisesRegex(KeyError, "Unit specified without value",
                               validateleaf, {'units': 'm'})

    def test_validate_dict_leaf_bad_two(self):
        self.assertRaisesRegex(KeyError, "No value", validateleaf,
                               {'Reference': 3})

    def test_is_leaf_num(self):
        self.assertTrue(is_leaf(3))

    def test_is_leaf_dict(self):
        self.assertTrue(is_leaf({'value': 3}))

    def test_is_leaf_dict_array(self):
        self.assertTrue(is_leaf({'value': [1, 2, 3], 'units': 'm'}))

    def test_is_leaf_dict_subdict(self):
        self.assertTrue(is_leaf({'value': {'a': 1, 'b': 2}, 'units': 'm'}))

    def test_is_leaf_dict_reference(self):
        self.assertTrue(
            is_leaf({
                'value': {
                    'a': 1,
                    'b': 2
                },
                'units': 'm',
                'reference': 'asdf'
            }))

    def test_not_is_leaf_dict_subdict(self):
        self.assertFalse(is_leaf({'a': {'value': 1}, 'reference': 'asdf'}))

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
        prob.model.add_constraint('con2', lower=0)
        prob.model.add_constraint('con3', lower=0)

        prob.setup()

        load_yaml_to_problem(prob, yaml_here_string)

        prob.run_driver()
        assert_near_equal(prob['B0'][0], 2.094, 1e-3)
        assert_near_equal(prob['B_on_coil'][0], 18, 1e-3)
        assert_near_equal(prob['n_coil'], 18, 1e-3)
        assert_near_equal(prob['geometry.r_ot'][0], 0.405, 1e-3)
        assert_near_equal(prob['windingpack.j_eff_max'][0], 160, 1e-3)


if __name__ == '__main__':
    unittest.main()
