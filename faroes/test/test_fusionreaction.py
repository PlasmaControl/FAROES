import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.fusionreaction
import faroes.units

import unittest


class TestSimpleFusionRateCoefficient(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.fusionreaction.SimpleRateCoeff()

        prob.setup(force_alloc_complex=True)
        prob.set_val("T", 10., units="keV")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_vals(self):
        prob = self.prob
        prob.run_driver()
        ratecoeff = prob.get_val('<σv>', units="m**3/s")
        assert_near_equal(ratecoeff, 1.1e-22)


class TestNBIBeamTargetFusion(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.fusionreaction.NBIBeamTargetFusion()

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_NBI", 50, units="MW")
        prob.set_val("<T_e>", 9.20, units="keV")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        fusrate = prob.get_val("P_fus", units="MW")
        assert_near_equal(fusrate, 34.6, tolerance=1e-3)


class TestTotalDTFusionRate(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = faroes.fusionreaction.TotalDTFusionRate()
        prob.setup(force_alloc_complex=True)
        prob.set_val("P_fus_th", 10, units="MW")
        prob.set_val("P_fus_NBI", 5, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        tot = prob.get_val("P_fus", units="MW")
        assert_near_equal(tot, 15, tolerance=1e-3)
        alpha = prob.get_val("P_α", units="MW")
        assert_near_equal(alpha, 3, tolerance=1e-3)
        n = prob.get_val("P_n", units="MW")
        assert_near_equal(n, 12, tolerance=1e-3)


class TestVolumetricThermalFusionRate(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.fusionreaction.VolumetricThermalFusionRate()

        prob.setup(force_alloc_complex=True)
        prob.set_val("<σv>", 1.1e-24, units="m**3/s")
        prob.set_val("n_D", 0.5, units="n20")
        prob.set_val("n_T", 0.5, units="n20")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        fusrate = prob.get_val("P_fus/V", units="W/m**3")
        assert_near_equal(fusrate, 7750, tolerance=1e-3)


class TestSimpleFusionAlphaSource(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.fusionreaction.SimpleFusionAlphaSource()

        prob.setup(force_alloc_complex=True)
        prob.set_val("rate", 1.0, units="mmol/s")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        fusrate = prob.get_val("S", units="1/s")
        assert_near_equal(fusrate, 6.022e20, tolerance=1e-3)


if __name__ == '__main__':
    unittest.main()
