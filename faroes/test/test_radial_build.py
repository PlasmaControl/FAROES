import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.radialbuild
from faroes.configurator import UserConfigurator

import unittest


class TestMenardSTInboardRadialBuild(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = faroes.radialbuild.MenardSTInboardRadialBuild(config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('plasma R_min', 2.4)

        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMenardSTOutboardRadialBuild(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model = faroes.radialbuild.MenardSTOutboardRadialBuild(config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('plasma R_max', 4.4)

        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
