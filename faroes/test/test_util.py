import faroes.util as util

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import numpy as np

import unittest


class TestDoubleSmoothShiftedReLu(unittest.TestCase):
    def setUp(self):
        dssrl = util.DoubleSmoothShiftedReLu(sharpness=25,
                                             x0=1.8,
                                             x1=2.25,
                                             s1=0.5,
                                             s2=0.1,
                                             units_out="m")
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
        dssrl = util.DoubleSmoothShiftedReLu(sharpness=25,
                                             x0=1.8,
                                             x1=2.25,
                                             s1=0.5,
                                             s2=0.1)
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


class TestOffsetCurveWithLimiter(unittest.TestCase):
    def setUp(self):
        x = [2, 2, 1, 1]
        y = [0, 1, 1, 0]
        θ_o = [-np.pi / 4, 1 * np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]

        opc = util.OffsetCurveWithLimiter()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units="m")
        ivc.add_output("y", val=y, units="m")
        ivc.add_output("θ_o", val=θ_o)
        ivc.add_output("x_min", val=0.5, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("opc", opc, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("s", 2**(1 / 2), units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check, atol=2e-5, rtol=2e-5)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        x_o = prob.get_val("opc.x_o")
        expected = [3, 3, 0.5, 0.5]
        assert_near_equal(x_o, expected, tolerance=1e-8)
        y_o = prob.get_val("opc.y_o")
        expected = [-1, 2, 1.5, -0.5]
        assert_near_equal(y_o, expected, tolerance=1e-8)


class TestOffsetCurveWithLimiter2(unittest.TestCase):
    def setUp(self):
        x = [4, 4, 3, 3]
        y = [0, 1, 1, 0]
        θ_o = [-np.pi / 4, 1 * np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]

        opc = util.OffsetCurveWithLimiter()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units="m")
        ivc.add_output("y", val=y, units="m")
        ivc.add_output("θ_o", val=θ_o)
        ivc.add_output("x_min", val=0.5, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("opc", opc, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("s", 2**(1 / 2), units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check, atol=2e-5, rtol=2e-5)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        x_o = prob.get_val("opc.x_o")
        expected = [5, 5, 2, 2]
        assert_near_equal(x_o, expected, tolerance=1e-8)
        y_o = prob.get_val("opc.y_o")
        expected = [-1, 2, 2, -1]
        assert_near_equal(y_o, expected, tolerance=1e-8)


class TestOffsetCurveWithLimiter3(unittest.TestCase):
    def setUp(self):
        x = np.array([
            10.51731228, 10.3315235, 9.81106632, 9.05371789, 8.1861703,
            7.32885186, 6.57185004, 5.96754431, 5.53680138, 5.28156277,
            5.19731228, 5.28156277, 5.53680138, 5.96754431, 6.57185004,
            7.32885186, 8.1861703, 9.05371789, 9.81106632, 10.3315235
        ])
        y = np.array([
            0.00000000e+00, 1.47299749e+00, 2.80180772e+00, 3.85635749e+00,
            4.53342012e+00, 4.76672000e+00, 4.53342012e+00, 3.85635749e+00,
            2.80180772e+00, 1.47299749e+00, 5.83754839e-16, -1.47299749e+00,
            -2.80180772e+00, -3.85635749e+00, -4.53342012e+00, -4.76672000e+00,
            -4.53342012e+00, -3.85635749e+00, -2.80180772e+00, -1.47299749e+00
        ])
        θ_o = np.array([
            0., 0.25100785, 0.49809132, 0.75850899, 1.08691177, 1.57079633,
            2.16398779, 2.60885909, 2.86614669, 3.02350726, 3.14159265,
            -3.02350726, -2.86614669, -2.60885909, -2.16398779, -1.57079633,
            -1.08691177, -0.75850899, -0.49809132, -0.25100785
        ])

        opc = util.OffsetCurveWithLimiter()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units="m")
        ivc.add_output("y", val=y, units="m")
        ivc.add_output("θ_o", val=θ_o)
        ivc.add_output("x_min", val=3.6969, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("opc", opc, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("s", 1.81462, units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check, atol=2e-5, rtol=2e-5)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        x_o = prob.get_val("opc.x_o")
        expected = np.array([
            12.3319246, 12.08927056, 11.40519605, 10.37087673, 9.03036655,
            7.32885186, 5.55746342, 4.40439744, 3.79254135, 3.69697368,
            3.69692728, 3.69697368, 3.79254135, 4.40439744, 5.55746342,
            7.32885186, 9.03036655, 10.37087673, 11.40519605, 12.08927056
        ])
        assert_near_equal(x_o, expected, tolerance=2e-6)
        y_o = prob.get_val("opc.y_o")
        expected = np.array([
            0.00000000e+00, 1.92371153e+00, 3.66873810e+00, 5.10452033e+00,
            6.13970483e+00, 6.58133232e+00, 6.03802561e+00, 4.77798092e+00,
            3.29478818e+00, 1.66098892e+00, 7.67499007e-16, -1.66098892e+00,
            -3.29478818e+00, -4.77798092e+00, -6.03802561e+00, -6.58133232e+00,
            -6.13970483e+00, -5.10452033e+00, -3.66873810e+00, -1.92371153e+00
        ])
        assert_near_equal(y_o, expected, tolerance=2e-6)


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


class TestSoftCapUnity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = util.SoftCapUnity()
        prob.setup(force_alloc_complex=True)
        prob.set_val('x', 1.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.set_val('x', 0.6)
        prob.run_driver()
        y = prob.get_val("y")
        assert_near_equal(y, 0.6, tolerance=1e-4)

    def test_values_2(self):
        prob = self.prob
        prob.set_val('x', 1.2)
        prob.run_driver()
        y = prob.get_val("y")
        assert (y < 1.0)


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        x = [0, 2, 3, 4]
        y = 2.5

        u = "m**3"  # arbitrary unit
        sm = util.Softmax(units=u)
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units=u)
        ivc.add_output("y", val=y, units=u)
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("sm", sm, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        self.prob = prob
        self.u = u

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        u = self.u
        z = prob.get_val("sm.z", units=u)
        expected = [2.5, 2.5, 3, 4]
        assert_near_equal(z, expected, tolerance=1e-6)


class TestPolygonalTorusVolume(unittest.TestCase):
    def setUp(self):
        x = [2, 2, 1, 1]
        y = [0, 1, 1, 0]

        ptv = util.PolygonalTorusVolume()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("R", val=x, units="m")
        ivc.add_output("Z", val=y, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("ptv", ptv, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check, atol=2e-5, rtol=2e-5)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        V = prob.get_val("ptv.V")
        expected = 9.424777961
        assert_near_equal(V, expected, tolerance=1e-8)


if __name__ == "__main__":
    unittest.main()
