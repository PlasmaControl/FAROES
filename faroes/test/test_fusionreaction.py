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


if __name__ == '__main__':
    unittest.main()
