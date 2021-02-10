import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.plasmacontrolsystem

import unittest


class TestSimplePlasmaControlPower(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasmacontrolsystem.SimplePlasmaControlPower()

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
        P_c = prob.get_val("P_control", units="MW")
        assert_near_equal(P_c, 4, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
