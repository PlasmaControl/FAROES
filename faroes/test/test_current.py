import faroes.current
import faroes.units  # noqa: F401

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import openmdao.api as om

import unittest


class TestQCylindrical(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.current.QCylindrical()

        prob.setup(force_alloc_complex=True)
        prob.set_val("Bt", 2.094, units="T")
        prob.set_val("Ip", 14.67, units="MA")
        prob.set_val("a", 1.875, units="m")
        prob.set_val("L_pol", 23.3, units="m")
        prob.set_val("R0", 3.0, units="m")

        prob.run_driver()
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(check)

    def test_vals(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["I/aB"], 3.74, tolerance=1e-2)
        assert_near_equal(prob["q_star"], 3.27, tolerance=1e-2)


class TestLineAveragedDensity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.current.LineAveragedDensity()

        prob.setup(force_alloc_complex=True)
        prob.set_val("Ip", 14.67, units="MA")
        prob.set_val("a", 1.875, units="m")
        prob.set_val("Greenwald fraction", 0.8)

        prob.run_driver()
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(check)

    def test_vals(self):
        prob = self.prob

        prob.run_driver()
        n_GW = prob.get_val("n_GW", units="n20")
        n_bar = prob.get_val("n_bar", units="n20")
        assert_near_equal(n_GW, 1.328, tolerance=1e-2)
        assert_near_equal(n_bar, 1.06, tolerance=1e-2)


class TestTotalPlasmaCurrent(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.current.TotalPlasmaCurrent()

        prob.setup(force_alloc_complex=True)
        prob.set_val("I_BS", 5.00, units="MA")
        prob.set_val("I_NBI", 5.00, units="MA")

        prob.run_driver()
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(check)

    def test_vals(self):
        prob = self.prob

        prob.run_driver()
        Ip = prob.get_val("Ip", units="MA")
        f_BS = prob.get_val("f_BS")
        assert_near_equal(Ip, 10, tolerance=1e-2)
        assert_near_equal(f_BS, 0.5, tolerance=1e-2)


if __name__ == "__main__":
    unittest.main()
