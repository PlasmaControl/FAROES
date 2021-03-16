import faroes.trappedparticles as trappedparticles

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import unittest


class TestTrappedParticleFractionUpperEst(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = trappedparticles.TrappedParticleFractionUpperEst()

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


class TestSauterTrappedParticleFractionCalc(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = trappedparticles.SauterTrappedParticleFractionCalc()

        prob.setup(force_alloc_complex=True)
        self.prob = prob

    def test_partials_positive(self):
        prob = self.prob
        prob.set_val('ε', 0.3)
        prob.set_val('δ', 0.2)
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_partials_negative(self):
        prob = self.prob
        prob.set_val('ε', 0.3)
        prob.set_val('δ', -0.2)
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
