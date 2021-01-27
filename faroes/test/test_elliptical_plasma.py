import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from faroes.configurator import UserConfigurator
import faroes.elliptical_plasma

import unittest


class TestPlasmaGeometry(unittest.TestCase):
    def test_partials(self):
        uc = UserConfigurator()
        prob = om.Problem()

        prob.model = faroes.elliptical_plasma.PlasmaGeometry(config=uc)
        #        prob.model.kappa_multiplier = 0.95
        #        prob.model.κ_ε_scaling_constants = [1.9, 1.9, 1.4]

        prob.setup()
        prob.set_val('A', 1.6)
        prob.set_val('R0', 3.0)

        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check)


class TestKappaScaling(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = faroes.elliptical_plasma.KappaScaling()
        prob.model.kappa_multiplier = 0.95
        prob.model.κ_ε_scaling_constants = [1.9, 1.9, 1.4]
        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
