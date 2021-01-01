import faroes.plasmaformulary

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


if __name__ == '__main__':
    unittest.main()
