import faroes.plasmaformulary
import faroes.units  # noqa: F401

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import numpy as np

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


class TestAverageIonMass(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model.add_subsystem('ivc',
                                 om.IndepVarComp('ni',
                                                 val=np.ones(2),
                                                 units='n20'),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem('aim',
                                 faroes.plasmaformulary.AverageIonMass(),
                                 promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("ni", np.array([0.53e20, 0.53e20]), units='m**-3')
        prob.set_val("Ai", [2, 3], units='u')

        self.prob = prob

    def test_value(self):
        prob = self.prob
        expected_A_bar = 2.5  # seconds
        prob.run_driver()
        assert_near_equal(prob["aim.A_bar"], expected_A_bar, tolerance=1e-3)

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

class TestCoulombLogarithmElectrons(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.plasmaformulary.CoulombLogarithmElectrons()

        prob.setup(force_alloc_complex=True)

        prob.set_val('ne', 1.06e20, units="m**-3")
        prob.set_val('Te', 9.20, units="keV")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["logΛe"], 17.372, tolerance=1e-5)

class TestCoulombLogarithmIons(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = faroes.plasmaformulary.CoulombLogarithmIons()
        prob.setup(force_alloc_complex=True)

        prob.set_val('ni', 0.80e20, units="m**-3")
        prob.set_val('Ti', 10.20, units="keV")
        prob.set_val('Z', 1.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["logΛi"], 20.34, tolerance=3e-3)

if __name__ == '__main__':
    unittest.main()
