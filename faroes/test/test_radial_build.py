import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.radialbuild as rb
from faroes.configurator import UserConfigurator

import unittest


class TestMenardSTOuterMachineRadialBuild(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model = rb.MenardSTOuterMachineRadialBuild()
        prob.setup(force_alloc_complex=True)
        prob.set_val('TF-cryostat thickness', 2.0)
        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMenardSTInboardRadialBuild(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = rb.MenardSTInboardRadialBuild(config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('CS R_max', 0.2)
        prob.set_val('TF R_min', 1.0)

        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMenardSTOutboardRadialBuild(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = rb.MenardSTOutboardRadialBuild(config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('plasma R_max', 4.4)

        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
