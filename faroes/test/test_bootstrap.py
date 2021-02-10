import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.bootstrap as bootstrap

import unittest


class TestBootstrapMultiplier(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = bootstrap.BootstrapMultiplier()

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

        prob.model = bootstrap.BootstrapFraction()

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


class TestBootstrapCurrent(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = bootstrap.BootstrapCurrent()
        prob.setup(force_alloc_complex=True)

        prob.set_val('ε', 1 / 1.6)
        prob.set_val('βp', 1.3571)
        prob.set_val('βp_th', 1.2098)
        prob.set_val('thermal pressure fraction', 0.89)
        prob.set_val('q_min', 2.2)
        prob.set_val('q_star', 3.56)
        prob.set_val('Ip', 14.67, units="MA")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        f_BS = prob.get_val("f_BS")
        expected = 0.7545
        assert_near_equal(f_BS, expected, tolerance=1e-2)


if __name__ == "__main__":
    unittest.main()
