import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.powerplant

import unittest


class TestPowerplantQ(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.powerplant.PowerplantQ()

        prob.setup(force_alloc_complex=True)

        prob.set_val("P_gen", 100, units="MW")
        prob.set_val("P_recirc", 50, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        Q = prob.get_val("Q_eng")
        assert_near_equal(Q, 2, tolerance=1e-4)


class TestAuxilliaryPower(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.powerplant.AuxilliaryPower()

        prob.setup(force_alloc_complex=True)

        prob.set_val("P_RF", 100, units="MW")
        prob.set_val("P_NBI", 50, units="MW")
        prob.set_val("η_RF", 0.5)
        prob.set_val("η_NBI", 0.5)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        P_aux = prob.get_val("P_aux", units="MW")
        P_auxe = prob.get_val("P_aux,e", units="MW")
        assert_near_equal(P_aux, 150, tolerance=1e-4)
        assert_near_equal(P_auxe, 300, tolerance=1e-4)


class TestTotalThermalPower(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.powerplant.TotalThermalPower()

        prob.setup(force_alloc_complex=True)

        prob.set_val("P_blanket", 100, units="MW")
        prob.set_val("P_α", 50, units="MW")
        prob.set_val("P_coolant", 10, units="MW")
        prob.set_val("P_aux", 10, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        P_priheat = prob.get_val("P_primary_heat", units="MW")
        P_heat = prob.get_val("P_heat", units="MW")
        assert_near_equal(P_priheat, 160, tolerance=1e-4)
        assert_near_equal(P_heat, 170, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
