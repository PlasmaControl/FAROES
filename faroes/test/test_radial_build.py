import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.radialbuild as rb
from faroes.configurator import UserConfigurator

import unittest


class TestMenardSTInboard(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = rb.MenardSTInboard(config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('TF R_out', 1.0)

        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMenardSTOutboard(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = rb.MenardSTOutboard()

        prob.setup(force_alloc_complex=True)

        prob.set_val('Ob FW R_in', 4.4)

        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMenardSTOuterMachine(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model = rb.MenardSTOuterMachine()
        prob.setup(force_alloc_complex=True)
        prob.set_val('TF-cryostat thickness', 2.0)
        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
