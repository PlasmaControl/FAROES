import faroes.util as util

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import numpy as np

import unittest


class TestDoubleSmoothShiftedReLu(unittest.TestCase):
    def setUp(self):
        dssrl = util.DoubleSmoothShiftedReLu(sharpness=25, x0=1.8, x1=2.25,
                                             s1=0.5, s2=0.1, units_out="m")
        prob = om.Problem()
        prob.model = dssrl
        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_values(self):
        prob = self.prob
        prob.set_val("x", 2)
        prob.run_driver()
        expected = 0.100103
        y = prob.get_val("y", units="m")
        assert_near_equal(y, expected, tolerance=1e-5)

        prob.set_val("x", 3)
        prob.run_driver()
        expected = 0.3
        y = prob.get_val("y", units="m")
        assert_near_equal(y, expected, tolerance=1e-5)

class TestDoubleSmoothShiftedReLuNoUnits(unittest.TestCase):
    def setUp(self):
        dssrl = util.DoubleSmoothShiftedReLu(sharpness=25, x0=1.8, x1=2.25,
                                             s1=0.5, s2=0.1)
        prob = om.Problem()
        prob.model = dssrl
        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_values(self):
        prob = self.prob
        prob.set_val("x", 2)
        prob.run_driver()
        expected = 0.100103
        y = prob.get_val("y")
        assert_near_equal(y, expected, tolerance=1e-5)

        prob.set_val("x", 3)
        prob.run_driver()
        expected = 0.3
        y = prob.get_val("y")
        assert_near_equal(y, expected, tolerance=1e-5)

class TestPolarAngleAndDistanceFromPoint(unittest.TestCase):
    def setUp(self):
        x = [1, 1, 0, 0]
        y = [0, 1, 1, 0]

        opc = util.PolarAngleAndDistanceFromPoint()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units="m")
        ivc.add_output("y", val=y, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("opc", opc, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("X0", 0.5)
        prob.set_val("Y0", 0.5)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        d_sq = prob.get_val("opc.d_sq", units="m**2")
        expected = [0.5, 0.5, 0.5, 0.5]
        assert_near_equal(d_sq, expected)
        θ = prob.get_val("opc.θ")
        expected = [-np.pi / 4, np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]
        assert_near_equal(θ, expected)


class TestOffsetParametricCurvePoints(unittest.TestCase):
    def setUp(self):
        x = [1, 1, 0, 0]
        y = [0, 1, 1, 0]
        dx_dt = [1, -1, -1, 1]
        dy_dt = [1, 1, -1, -1]

        opc = util.OffsetParametricCurvePoints()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units="m")
        ivc.add_output("dx_dt", val=dx_dt, units="m")
        ivc.add_output("y", val=y, units="m")
        ivc.add_output("dy_dt", val=dy_dt, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("opc", opc, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("s", 2**(1 / 2), units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()


class TestSmoothShiftedReLu(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = util.SmoothShiftedReLu(x0=1, bignum=20)

        prob.setup(force_alloc_complex=True)
        prob.set_val("x", 1.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        expected = 0.0346574
        y = prob.get_val("y")
        assert_near_equal(y, expected, tolerance=1e-4)


if __name__ == "__main__":
    unittest.main()
