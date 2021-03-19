import faroes.units  # noqa: F401
import faroes.confinementtime
from faroes.configurator import UserConfigurator

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import unittest
from importlib import resources


class TestConfinementTime(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = faroes.confinementtime.ConfinementTime(config=uc)
        prob.setup()
        prob.set_val('H', 2.0)
        prob.set_val('Ip', 14.67, units='MA')
        prob.set_val('Bt', 2.094, units='T')
        prob.set_val('n19', 10.63, units="n19")
        prob.set_val('PL', 83.34, units="MW")
        prob.set_val('R', 3.0, units='m')
        prob.set_val('ε', 1 / 1.6)
        prob.set_val('κa', 2.19)
        prob.set_val('M', 2.5)
        self.prob = prob

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val('τe'), 2.76, tolerance=1e-2)


class TestConfinementTimeScaling(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = faroes.confinementtime.ConfinementTimeScaling(config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('R', 1.3)
        prob.set_val('κa', 2.73977961)
        prob.set_val('Ip', 14)
        prob.set_val('ε', 0.2)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

        prob.run_driver()

    def test_user_fewer(self):
        prob = om.Problem()

        resource_dir = 'faroes.test.test_data'
        resource_name = 'confinementtime.yaml'
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = faroes.confinementtime.ConfinementTimeScaling(
            config=uc, scaling='User')

        prob.setup(force_alloc_complex=True)

        prob.set_val('Bt', 10.0, units="T")
        prob.set_val('Ip', 10, units="MA")

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

        prob.run_driver()
        assert_near_equal(prob.get_val('τe'), 1.0)

    def test_user_extra(self):
        prob = om.Problem()

        resource_dir = 'faroes.test.test_data'
        resource_name = 'confinementtime.yaml'
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        with self.assertRaisesRegex(ValueError, "Unknown term"):
            prob.model = faroes.confinementtime.ConfinementTimeScaling(
                config=uc, scaling='MAST-MG')
            prob.setup(force_alloc_complex=True)


class TestHybridConfinementTime(unittest.TestCase):
    def test_menard_T(self):
        # from the column of "T" cells
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = faroes.confinementtime.MenardHybridScaling(config=uc)

        prob.setup()

        prob.set_val('Ip', 14.67, units='MA')
        prob.set_val('Bt', 2.094, units='T')
        prob.set_val('n19', 10.63, units='n19')
        prob.set_val('PL', 83.34)
        prob.set_val('R', 3.0, units='m')
        prob.set_val('ε', 1 / 1.6)
        prob.set_val('κa', 2.19)
        prob.set_val('M', 2.5)
        prob.run_driver()
        assert_near_equal(prob.get_val('τe'), 1.6, tolerance=0.01)


if __name__ == '__main__':
    unittest.main()
