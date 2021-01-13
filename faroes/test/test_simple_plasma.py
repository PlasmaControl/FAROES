import faroes.simple_plasma

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

from scipy.constants import electron_mass

import unittest


class TestMainIonMix(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.MainIonMix()

        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestIonMixMux(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.IonMixMux()

        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()


class TestThermalVelocity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.ThermalVelocity(mass=electron_mass)
        prob.setup(force_alloc_complex=True)
        prob.set_val('T', 10, units='keV')
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestZeroDPlasmaDensities(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.ZeroDPlasmaDensities()

        prob.setup(force_alloc_complex=True)
        prob.set_val('Z_imp', 6)
        prob.set_val('Z_eff', 2)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestZeroDPlasmaStoredEnergy(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.ZeroDPlasmaStoredEnergy()

        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestZeroDPlasmaPressures(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = faroes.simple_plasma.ZeroDPlasmaPressures()
        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        prob.set_val("<p_th>", 300.16, units="kPa")
        prob.set_val("Z_ave", 1.20)
        prob.set_val("Ti/Te", 1.10)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.set_val("<p_th>", 300.16, units="kPa")
        prob.set_val("Z_ave", 1.20)
        prob.set_val("Ti/Te", 1.10)
        prob.run_driver()
        assert_near_equal(prob["<p_e>"], 156.60, tolerance=1e-3)
        assert_near_equal(prob["<p_i>"], 143.55, tolerance=1e-3)


class TestZeroDPlasmaTemperatures(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.ZeroDPlasmaTemperatures()

        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        prob.set_val("<n_e>", 1.06e20)
        prob.set_val("<p_e>", 156.6)
        prob.set_val("<p_i>", 143.55)
        prob.set_val("ni/ne", 0.83)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.set_val("<n_e>", 1.06e20)
        prob.set_val("<p_e>", 156.6)
        prob.set_val("<p_i>", 143.55)
        prob.set_val("ni/ne", 0.83)
        prob.run_driver()
        assert_near_equal(prob["<T_i>"], 10.12, tolerance=1e-2)
        assert_near_equal(prob["<T_e>"], 9.20, tolerance=1e-2)


class TestZeroDFusionPower(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.simple_plasma.ZeroDThermalFusionPower()

        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
