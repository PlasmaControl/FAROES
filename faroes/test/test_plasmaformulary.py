import faroes.plasmaformulary
from faroes.units import add_local_units
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import unittest


class TestAlfvenSpeed(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasmaformulary.AlfvenSpeed()

        prob.setup(force_alloc_complex=True)

        prob.set_val('|B|', 1.0)
        prob.set_val('ρ', 1.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["V_A"], 892.062, tolerance=1e-5)

class TestFastParticleHeatingFractions(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasmaformulary.FastParticleHeatingFractions()

        prob.setup(force_alloc_complex=True)

        prob.set_val('W/Wc', 1.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        prob.run_driver()
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        expected = (2/9) * (3**(1/2) * np.pi - 3 * np.log(2))
        assert_near_equal(prob["f_i"], expected, tolerance=1e-5)


class TestSlowingThermalizationTime(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasmaformulary.SlowingThermalizationTime()

        prob.setup(force_alloc_complex=True)

        prob.set_val('W/Wc', 1.0)
        prob.set_val('ts', 1.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["τth"], 0.231049, tolerance=1e-5)


class TestCriticalSlowingEnergy(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        add_local_units()

        prob.model.add_subsystem('ivc',
                                 om.IndepVarComp('ni',
                                                 val=np.ones(2),
                                                 units='n20'),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem(
            'cse',
            faroes.plasmaformulary.CriticalSlowingEnergy(),
            promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("At", 2, units='u')
        prob.set_val("ne", 1.0e20, units='m**-3')
        prob.set_val("Te", 1.0, units='keV')
        prob.set_val("ni", np.array([0.5e20, 0.5e20]), units='m**-3')
        prob.set_val("Ai", [2, 3], units='u')
        prob.set_val("Zi", [1, 1])
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.set_val("At", 2, units='u')
        prob.set_val("ne", 1.0e20, units='m**-3')
        prob.set_val("Te", 1.0, units='keV')
        prob.set_val("ni", np.array([0.5e20, 0.5e20]), units='m**-3')
        prob.set_val("Ai", [2, 3], units='u')
        prob.set_val("Zi", [1, 1])
        prob.run_driver()
        assert_near_equal(prob["cse.W_crit"], 27.872, tolerance=1e-3)


if __name__ == '__main__':
    unittest.main()
