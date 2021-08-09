import faroes.shapefactor as sf
import openmdao.api as om
import unittest
import math
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal


class TestShapeFactorConst(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = sf.ConstProfile()

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("δ0", 0)
        prob.set_val("κ", 1)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs', step=1e-10)
        assert_check_partials(check)

        prob.run_driver()

    def test_volume(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val("S"), 2 * math.pi**2 * 5/2)


class TestShapeFactorParabConst(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = sf.ParabProfileConstTriang()

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("δ0", 0.2)
        prob.set_val("κ", 1)
        prob.set_val("α", 2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs', step=1e-10)
        assert_check_partials(check)

        prob.run_driver()

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val("S"), 16.14463285, tolerance=1e-4)


class TestShapeFactorParabLinear(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = sf.ParabProfileLinearTriang()

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("δ0", 0.2)
        prob.set_val("κ", 1)
        prob.set_val("α", 2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='fd',
                                    step=1e-4, form="central")
        assert_check_partials(check, atol=3.5e-05, rtol=1e-05)

        prob.run_driver()

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val("S"), 16.24507658, tolerance=1e-4)


if __name__ == '__main__':
    unittest.main()
