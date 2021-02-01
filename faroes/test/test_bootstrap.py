import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.bootstrap

import unittest


class TestBootstrapMultiplier(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.bootstrap.BootstrapMultiplier()

        prob.setup(force_alloc_complex=True)

        prob.set_val("q_star", 1.0)
        prob.set_val("q_min", 1.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["bs_mult"], 1.00086, tolerance=1e-4)


class TestBootstrapFraction(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.bootstrap.BootstrapFraction()

        prob.setup(force_alloc_complex=True)

        prob.set_val("bs_mult", 1.0)
        prob.set_val("βp_th", 1.0)
        prob.set_val("ε", 0.36)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["f_BS"], 0.54, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
