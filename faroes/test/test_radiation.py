import faroes.radiation

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import openmdao.api as om

import unittest


class TestTrivialRadiation(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.radiation.TrivialRadiation()

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_heat", 100, units="MW")
        prob.set_val("f_rad", 0.3)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(check)

    def test_vals(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["P_loss"], 70, tolerance=1e-3)
        assert_near_equal(prob["P_rad"], 30, tolerance=1e-3)


if __name__ == "__main__":
    unittest.main()
