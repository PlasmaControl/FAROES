import faroes.nbicd
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials
import numpy as np

import openmdao.api as om

import unittest

class TestCurrentDriveAlphaCubed(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model.add_subsystem('ivc',
                                 om.IndepVarComp('ni',
                                                 val=np.ones(2),
                                                 units='n20'),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem(
            'cda',
            faroes.nbicd.CurrentDriveAlphaCubed(),
            promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("ni", np.array([0.53e20, 0.53e20]), units='m**-3')
        prob.set_val("Ai", [2, 3], units='u')
        prob.set_val("Zi", [1, 1])

        prob.set_val("ne", 1.06e20, units='m**-3')
        prob.set_val("ve", 40250, units='km/s')
        prob.set_val("v0", 6922, units='km/s')
        self.prob = prob

    def test_value(self):
        prob = self.prob
        expectedτth = 0.05974  # seconds
        prob.run_driver()
        assert_near_equal(prob["cda.α³"], expectedτth, tolerance=1e-3)

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

class TestCurrentDriveA(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.CurrentDriveA()

        prob.setup(force_alloc_complex=True)
        prob.set_val('vb', 10)
        prob.set_val('vth_e', 60)
        prob.set_val('Z_eff', 1.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["A"], 10/7, tolerance=1e-3)

class TestCurrentDriveIntegral(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.CurrentDriveIntegral()

        prob.setup(force_alloc_complex=True)
        prob.set_val('β1', 2.5)
        prob.set_val('α³', 0.32)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, rtol=1e-5)

    def test_values(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["i"], 0.307878, tolerance=1e-3)


class TestTrappedParticleFractionUpperEst(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.TrappedParticleFractionUpperEst()

        prob.setup(force_alloc_complex=True)
        prob.set_val('ε', 0.1)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["ftrap_u"], 0.455802, tolerance=1e-4)


if __name__ == '__main__':
    unittest.main()
