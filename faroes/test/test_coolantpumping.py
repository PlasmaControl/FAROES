import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.coolantpumping

import unittest


class TestSimpleCoolantPumpingPower(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.coolantpumping.SimpleCoolantPumpingPower()

        prob.setup(force_alloc_complex=True)

        prob.set_val("P_thermal", 100, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        P_p = prob.get_val("P_pumps", units="MW")
        assert_near_equal(P_p, 3, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
