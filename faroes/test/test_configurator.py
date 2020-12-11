from openmdao.utils.assert_utils import assert_near_equal
from faroes.configurator import UserConfigurator
import unittest
from ruamel.yaml import YAML
from importlib import resources


class TestConfigurator(unittest.TestCase):
    def test_create_configurator(self):
        UserConfigurator()


class TestWalkYamlTree(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()

    def test_walk_yaml_tree(self):
        yaml = YAML(typ='safe')
        tree = """
        myprop:
          a: 3
          b: {value: 1.0, units: m}
          c: astring
        """
        data = yaml.load(tree)
        ret = self.uc._walk_yaml_tree(data)
        self.assertIsNone(ret)


class TestIsLeaf(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._is_leaf

    def test_1(self):
        self.assertTrue(self.f(1))

    def test_1_point_0(self):
        self.assertTrue(self.f(1.0))

    def test_default(self):
        self.assertTrue(self.f("default"))

    def test_rando(self):
        # it _is_ a leaf, just a potentially invalid one
        self.assertTrue(self.f("random_Text"))

    def test_false_one(self):
        self.assertFalse(self.f({"thing": "A"}))

    def test_dict_okay(self):
        self.assertTrue(self.f({"value": 1}))

    def test_dict_okay_2(self):
        self.assertTrue(self.f({"value": 1, "units": "m"}))

    def test_dict_bad_3(self):
        self.assertFalse(self.f({"units": "m"}))

    def test_dict_okay_3(self):
        self.assertTrue(self.f({"value": "default", "units": "m"}))

    def test_dict_array(self):
        self.assertTrue(self.f({'value': [1, 2, 3], 'units': 'm'}))

    def test_dict_subdict(self):
        self.assertTrue(self.f({'value': {'a': 1, 'b': 2}, 'units': 'm'}))


class TestValidateNumber(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._validate_number

    def test(self):
        res = self.f(1)
        self.assertIsNone(res)

    def test_unit_error(self):
        with self.assertRaises(AttributeError):
            self.f(1, units='m')

    def test_unit_none(self):
        self.f(1, units=None)

    def test_unit_ignore(self):
        self.f(1, units=self.uc.ignore_units)


class TestValidateDictLeaf(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._validate_dict_leaf

    def test_okay(self):
        self.f({'value': 3, 'units': 'm'}, units='m')

    def test_okay_2(self):
        self.f({'value': 3, 'units': 'm', 'ref': 'hi'}, units='m')

    def test_okay_3(self):
        self.f({'value': 3, 'ref': 'hi'})

    def test_compat(self):
        self.f({'value': 3, 'units': 'm'}, units='mm')

    def test_list_okay(self):
        self.f({'value': [1, 2, 3]})

    def test_raises_if_extra(self):
        with self.assertRaisesRegex(KeyError, "Extra key"):
            self.f({'value': 3, 'units': 'm'})

    def test_raises_no_units(self):
        with self.assertRaisesRegex(KeyError, "No key"):
            self.f({'value': 3}, units='m')

    def test_raises_incompat(self):
        with self.assertRaisesRegex(AttributeError, "incompatible"):
            self.f({'value': 3, 'units': 'm'}, units='mT')

    def test_raises_bad_str(self):
        with self.assertRaisesRegex(AttributeError, "The default"):
            self.f({'value': "cow", 'units': 'm'}, units='m')

    def test_raises_no_value_w_units(self):
        with self.assertRaisesRegex(KeyError, "specified without value"):
            self.f({'units': 'm'}, units='m')

    def test_raises_no_value(self):
        with self.assertRaisesRegex(KeyError, "No value"):
            self.f({'ref': 'm'}, units='m')


class TestCrossValidateWithList(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._cross_validate_with_list

    def test_unequal(self):
        with self.assertRaisesRegex(AttributeError, "unequal length"):
            self.f([1, 2, 3], [1, 2])

    def test_default(self):
        self.f([1, 2, 3], "Default")

    def test_okay(self):
        self.f([1, 2, 3], [1, 2, 3])


class TestCrossValidateWithDict(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._cross_validate_with_dict

    def test_default(self):
        self.f({'value': 1, 'units': 'm'}, "Default")

    def test_no_dict(self):
        with self.assertRaisesRegex(AttributeError, "provides a dict"):
            self.f({'a': 1, 'b': 'm'}, 666)

    def test_extra_key(self):
        with self.assertRaisesRegex(KeyError, "not present in"):
            self.f({'a': 1, 'b': 'm'}, {'c': 3})

    def test_okay(self):
        self.f({'a': 1, 'b': 'm'}, {'a': 3})

    def test_okay_2(self):
        self.f({'a': 1, 'b': 'm'}, {'a': 3, 'b': 'cow'})


class TestCrossValidateLeafTypes(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._cross_validate_leaf_types

    def test_num(self):
        self.f(1, 2)

    def test_str(self):
        self.f("cow", "horse")

    def test_default(self):
        self.f("cow", "default")

    def test_bad(self):
        with self.assertRaisesRegex(TypeError, "unacceptable"):
            self.f({1, 2, 3}, {1, 2, 3})

    def test_unhandled(self):
        with self.assertRaisesRegex(TypeError, "Unhandled"):
            self.f({1, 2, 3}, 1)

    def test_dicts(self):
        self.f({'value': 1, 'units': 'm'}, {'value': 1, 'units': 'cm'})

    def test_bool(self):
        """Bool is a numbers.Number"""
        self.f(False, True)

    def test_not_a_dict(self):
        with self.assertRaisesRegex(TypeError, "not a dict"):
            self.f({'a': 1}, 3)

    def test_cross_validate_leaf_types_not_a_number(self):
        with self.assertRaisesRegex(TypeError, "not a number"):
            self.f(3, {'value': 1})

    def test_cross_validate_leaf_types_not_a_str(self):
        with self.assertRaisesRegex(TypeError, "not a string"):
            self.f("cow", 666)


class TestCrossValidateLeafDicts(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc._cross_validate_leaf_dicts

    def test_okay(self):
        self.f({'value': 1}, {'value': 2})

    def test_no_val(self):
        with self.assertRaisesRegex(KeyError, "No value"):
            self.f({'value': 1}, {'schmalue': 2})

    def test_units_okay(self):
        self.f({'value': 1, 'units': 'm'}, {'value': 2, 'units': 'cm'})

    def test_missing_u(self):
        with self.assertRaisesRegex(KeyError, "No key"):
            self.f({'value': 1, 'units': 'm'}, {'value': 2})

    def test_no_units(self):
        self.f({'value': 1}, {'value': 2})


class TestHandleLeafMatching(unittest.TestCase):
    def setUp(self):
        uc = UserConfigurator()
        self.f = uc._handle_leaf_matching

    def test_matching_numbers(self):
        d = {'a': 1}
        u = {'a': 2}
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k], u[k])

    def test_matching_strings(self):
        d = {'a': "cow"}
        u = {'a': "horse"}
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k], u[k])

    def test_nums_default(self):
        d = {'a': 3}
        u = {'a': "default"}
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k], 3)

    def test_handle_leaf_matching_strings_default(self):
        d = {'a': "cow"}
        u = {'a': "default"}
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k], "cow")

    def test_handle_leaf_matching_dicts(self):
        d = {'a': {'value': 3, 'units': 'm'}}
        u = {'a': {'value': 200, 'units': 'cm'}}
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k]['value'], 200)
        self.assertEqual(d[k]['units'], 'cm')

    def test_non_matching_units(self):
        d = {'a': {'value': 3, 'units': 'm'}}
        u = {'a': {'value': 200, 'units': 'T'}}
        k = 'a'
        with self.assertRaisesRegex(AttributeError, "incompatible"):
            self.f(d, u, k)

    def test_dict_default_value(self):
        d = {'a': {'value': 3, 'units': 'm'}}
        u = {'a': {'value': "default", 'units': 'm'}}
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k]['value'], 3)

    def test_dict_default_value_incompat(self):
        d = {'a': {'value': 3, 'units': 'm'}}
        u = {'a': {'value': "default", 'units': 'T'}}
        k = 'a'
        with self.assertRaisesRegex(AttributeError, "incompatible"):
            self.f(d, u, k)

    def test_dict_extra_key_transfer(self):
        d = {'a': {'value': 3, 'units': 'm', 'ref': 'B'}}
        u = {'a': {'value': "default", 'units': 'm', 'ref': 'A'}}
        k = 'a'
        self.f(d, u, k)
        self.assertTrue('ref' in d[k])
        self.assertEqual(d[k]['ref'], 'A')
        self.assertEqual(d[k]['value'], 3)

    def test_dict_extra_key_transfer_default_units(self):
        d = {'a': {'value': 3, 'units': 'm', 'keep': 'B', 'change': 'B'}}
        u = {
            'a': {
                'value': "default",
                'units': 'm',
                'keep': 'default',
                'change': 'A'
            }
        }
        k = 'a'
        self.f(d, u, k)
        self.assertEqual(d[k]['keep'], 'B')
        self.assertEqual(d[k]['change'], 'A')


class TestWalkTwoTrees(unittest.TestCase):
    def setUp(self):
        uc = UserConfigurator()
        self.yaml = YAML(typ='safe')
        self.f = uc._walk_two_yaml_trees

    def test_single_leaf(self):
        tree1 = """
        a: 3
        """
        self.n1 = self.yaml.load(tree1)
        n1 = self.n1
        tree = """
        a: 4
        """
        n2 = self.yaml.load(tree)
        self.f(n1, n2)
        self.assertEqual(n1['a'], 4)

    def test_extra_dict(self):
        tree1 = """
        a: 3
        """
        self.n1 = self.yaml.load(tree1)
        n1 = self.n1
        tree = """
        a: 4
        b: 5
        """
        n2 = self.yaml.load(tree)
        with self.assertRaisesRegex(KeyError, 'not present'):
            self.f(n1, n2)

    def test_simple_list(self):
        n1 = [1, 2]
        n2 = [1, 3]
        self.f(n1, n2)
        self.assertEqual(n1, [1, 3])

    def test_simple_list_default(self):
        n1 = [5, 2]
        n2 = ["default", 3]
        self.f(n1, n2)
        self.assertEqual(n1, [5, 3])

    def test_simple_list_bad(self):
        n1 = [1, 2]
        n2 = [1, "cow"]
        with self.assertRaisesRegex(TypeError, 'not a number'):
            self.f(n1, n2)

    def test_single_list_default(self):
        n1 = {'a': [1, 2]}
        n2 = {'a': "default"}
        self.f(n1, n2)
        self.assertEqual(n1['a'], [1, 2])

    def test_default_only(self):
        n1 = {'a': [1, 2]}
        n2 = "default"
        self.f(n1, n2)
        self.assertEqual(n1['a'], [1, 2])

    def test_complicated_default(self):
        n1 = {
            'magnets': [
                {
                    'r1': {
                        'value': 2,
                        'units': 'm'
                    }
                },
            ]
        }
        n2 = {'magnets': "default"}
        self.f(n1, n2)
        self.assertEqual(n1['magnets'][0]['r1']['value'], 2)

    def test_complicated_list_default(self):
        n1 = {
            'magnets': [
                {
                    'r1': {
                        'value': 2,
                        'units': 'm'
                    }
                },
            ]
        }
        n2 = {
            'magnets': [
                {
                    'r1': "default"
                },
            ]
        }
        self.f(n1, n2)
        self.assertEqual(n1['magnets'][0]['r1']['value'], 2)

    def test_complicated_exclude_refs(self):
        n1 = {
            'magnets': [
                {
                    'r1': {
                        'value': 2,
                        'units': 'm'
                    },
                    'references': ["ref1"]
                },
            ]
        }
        n2 = {
            'magnets': [
                {
                    'r1': "default"
                },
            ]
        }
        self.f(n1, n2, exclude_keys=["references"])
        self.assertEqual(n1['magnets'][0]['r1']['value'], 2)

    def test_complicated_exclude_refs_two(self):
        n1 = {
            'magnets': [
                {
                    'r1': {
                        'value': 2,
                        'units': 'm'
                    },
                    'references': ["ref1"]
                },
            ]
        }
        n2 = {
            'magnets': [
                {
                    'r1': {
                        'value': 3,
                        'units': 'm',
                        'references': 'test'
                    },
                },
            ],
            'references':
            'test'
        }
        self.f(n1, n2, exclude_keys=["references"])
        self.assertEqual(n1['magnets'][0]['r1']['value'], 3)


class TestGetValue(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc.get_value

    def test_with_units(self):
        ρ = self.f(('materials', 'stainless steel 316', 'density'),
                   units='kg/m**3')
        self.assertEqual(ρ, 7930)

    def test_int(self):
        n = self.f(('materials', 'lead', 'atomic number'))
        self.assertEqual(n, 82)

    def test_with_conv(self):
        ρ = self.f(('materials', 'stainless steel 316', 'density'),
                   units='g/cm**3')
        assert_near_equal(ρ, 7.930)

    def test_units_bad(self):
        with self.assertRaisesRegex(AttributeError, "incompatible"):
            self.f(('materials', 'stainless steel 316', 'density'), units='T')

    def test_units_none(self):
        with self.assertRaisesRegex(KeyError, "Extra key 'units'"):
            self.f(('materials', 'stainless steel 316', 'density'))

    def test_index(self):
        n = self.f(('materials', 'test_material', 0))
        self.assertEqual(n, 0)

    def test_index_dict(self):
        n = self.f(('materials', 'test_material', 3), 'm')
        self.assertEqual(n, 30)

    def test_list(self):
        n = self.f(('fits', 'marginal κ-ε scaling', 'constants'))
        self.assertEqual(n, [1.9, 1.9, 1.4])

    def test_index_list(self):
        n = self.f(('fits', 'marginal κ-ε scaling', 'constants'))
        self.assertEqual(n, [1.9, 1.9, 1.4])


class TestUpdateConfiguration(unittest.TestCase):
    def setUp(self):
        self.uc = UserConfigurator()
        self.f = self.uc.get_value
        self.resource_dir = 'faroes.test.test_data'

    def test(self):
        resource_name = 'config.yaml'
        if resources.is_resource(self.resource_dir, resource_name):
            with resources.path(self.resource_dir, resource_name) as path:
                self.uc.update_configuration(path)

    def test_bad_key(self):
        resource_name = 'config_bad_option.yaml'
        with self.assertRaisesRegex(KeyError, "not present in"):
            with resources.path(self.resource_dir, resource_name) as path:
                self.uc.update_configuration(path)

    def test_default_value(self):
        resource_name = 'config_default.yaml'
        with resources.path(self.resource_dir, resource_name) as path:
            self.uc.update_configuration(path)
        res = self.f(('magnet_geometry', 'inter-block clearance'), 'cm')
        assert_near_equal(res, 0.2)

    def test_full_default(self):
        with resources.path(self.resource_dir, 'config_default.yaml') as path:
            self.uc.update_configuration(path)
        res = self.f(('magnet_geometry', 'ground wrap thickness'), 'cm')
        assert_near_equal(res, 0.4)

    def test_full_normal(self):
        with resources.path(self.resource_dir, 'config_normal.yaml') as path:
            self.uc.update_configuration(path)
        res = self.f(('magnet_geometry', 'external structure thickness'), 'cm')
        assert_near_equal(res, 10)

    def test_okay_units(self):
        with resources.path(self.resource_dir,
                            'config_okay_units.yaml') as path:
            self.uc.update_configuration(path)
        res = self.f(('magnet_geometry', 'inter-block clearance'), 'mm')
        assert_near_equal(res, 1)


class TestUpdateConfigurationLoading(unittest.TestCase):
    def setUp(self):
        self.resource_dir = 'faroes.test.test_data'

    def test(self):
        with resources.path(self.resource_dir,
                            'config_okay_units.yaml') as path:
            self.uc = UserConfigurator(path)
        self.f = self.uc.get_value
        res = self.f(('magnet_geometry', 'inter-block clearance'), 'mm')
        assert_near_equal(res, 1)

    def test_repeat_same(self):
        with resources.path(self.resource_dir,
                            'config_okay_units.yaml') as path:
            self.uc = UserConfigurator(path)
            self.uc.update_configuration(path)
        self.f = self.uc.get_value
        res = self.f(('magnet_geometry', 'inter-block clearance'), 'mm')
        assert_near_equal(res, 1)

    def test_three(self):
        with resources.path(self.resource_dir,
                            'config_okay_units.yaml') as path:
            self.uc = UserConfigurator(path)
        with resources.path(self.resource_dir, 'config.yaml') as path:
            self.uc.update_configuration(path)
        self.f = self.uc.get_value
        res = self.f(('magnet_geometry', 'inter-block clearance'), 'mm')
        assert_near_equal(res, 2)


if __name__ == '__main__':
    unittest.main()
