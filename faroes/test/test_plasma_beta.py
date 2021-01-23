import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.plasma_beta

import unittest


class TestBetaNComputation(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasma_beta.BetaNTotal()
        prob.model.β_ε_scaling_constants = [3.12, 3.5, 1.7]
        prob.model.β_N_multiplier = 1.1

        prob.setup(force_alloc_complex=True)

        prob.set_val("A", 1.6)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["β_N"], 0.046942, tolerance=1e-4)
        assert_near_equal(prob["β_N total"], 0.0516, tolerance=1e-2)

class TestBetaToroidal(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasma_beta.BetaToroidal()

        prob.setup(force_alloc_complex=True)

        prob.set_val("Bt", 1, units="T")
        prob.set_val("Ip", 1, units="MA")
        prob.set_val("a", 1, units="m")
        prob.set_val("β_N total", 0.01, units="m * T / MA")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["βt"], 0.01, tolerance=1e-4)

class TestSpecifiedTotalAveragePressure(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasma_beta.SpecifiedTotalAveragePressure()

        prob.setup(force_alloc_complex=True)

        prob.set_val("Bt", 1, units="T")
        prob.set_val("βt", 0.01)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["<p_tot>"], 3978.87, tolerance=1e-4)

class TestBPoloidal(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasma_beta.BPoloidal()

        prob.setup(force_alloc_complex=True)

        prob.set_val("Ip", 1, units="MA")
        prob.set_val("L_pol", 6.28319, units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["Bp"], 1/5, tolerance=1e-4)

class TestBetaPoloidal(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasma_beta.BetaPoloidal()

        prob.setup(force_alloc_complex=True)

        prob.set_val("Bp", 1, units="T")
        prob.set_val("<p_tot>", 1, units="kPa")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["βp"], 0.00251327, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
