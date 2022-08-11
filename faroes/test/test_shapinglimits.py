import faroes.shapinglimits as sl

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import unittest


class TestKappaScaling(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = sl.MenardKappaScaling()
        prob.model.kappa_multiplier = 0.95
        prob.model.κ_ε_scaling_constants = [1.9, 1.9, 1.4]
        prob.model.κ_area_frac = 0.8
        prob.setup(force_alloc_complex=True)
        prob.set_val('A', 1.6)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestZohmMaximumKappaScaling(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = sl.ZohmMaximumKappaScaling()

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 2.6)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
