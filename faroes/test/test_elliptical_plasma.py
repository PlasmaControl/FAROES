import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.elliptical_plasma

import unittest


class TestBetaNComputation(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.elliptical_plasma.PlasmaBetaNTotal()
        prob.model.β_ε_scaling_constants = [3.12, 3.5, 1.7]
        prob.model.β_N_multiplier = 1.1

        prob.setup()

        prob.set_val('A', 1.6)

        prob.run_driver()
        assert_near_equal(prob["beta_N"], 4.6942, tolerance=1e-4)
        assert_near_equal(prob["beta_N total"], 5.16, tolerance=1e-2)


class TestPlasmaGeometry(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.elliptical_plasma.PlasmaGeometry()
        prob.model.kappa_multiplier = 0.95
        prob.model.κ_ε_scaling_constants = [1.9, 1.9, 1.4]

        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
