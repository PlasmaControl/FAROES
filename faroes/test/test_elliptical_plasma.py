from faroes.configurator import UserConfigurator
import faroes.elliptical_plasma as elliptical_plasma

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import unittest


class TestEllipseLikeGeometry(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = elliptical_plasma.EllipseLikeGeometry()

        prob.setup(force_alloc_complex=True)
        prob.set_val('A', 1.6)
        prob.set_val('a', 1.875)
        prob.set_val('R0', 3.0)
        prob.set_val('κ', 2.7)
        prob.set_val('κa', 2.7)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        prob.run_driver()
        prob.model.set_check_partial_options('*', method='fd')
        check = prob.check_partials(out_stream=None)
        assert_check_partials(check, atol=1e-3)


class TestMenardPlasmaGeometry(unittest.TestCase):
    def setUp(self):
        uc = UserConfigurator()
        prob = om.Problem()

        prob.model = elliptical_plasma.MenardPlasmaGeometry(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val('A', 1.6)
        prob.set_val('a', 1.875)
        prob.set_val('R0', 3.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, atol=1e-3)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        V = prob.get_val("V", units="m**3")
        expected = 456
        assert_near_equal(V, expected, tolerance=3e-3)
        S_c = prob.get_val("S_c", units="m**2")
        expected = 24
        assert_near_equal(S_c, expected, tolerance=1e-2)
        L_pol = prob.get_val("L_pol", units="m")
        expected = 23.2
        assert_near_equal(L_pol, expected, tolerance=1e-2)


class TestEllipticalPlasmaGeometry(unittest.TestCase):
    def setUp(self):
        uc = UserConfigurator()
        prob = om.Problem()

        prob.model = elliptical_plasma.EllipticalPlasmaGeometry(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val('A', 1.6)
        prob.set_val('a', 1.875)
        prob.set_val('R0', 3.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, atol=1e-3)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        V = prob.get_val("V", units="m**3")
        expected = 570
        assert_near_equal(V, expected, tolerance=3e-3)
        S_c = prob.get_val("S_c", units="m**2")
        expected = 30
        assert_near_equal(S_c, expected, tolerance=1e-2)


if __name__ == '__main__':
    unittest.main()
