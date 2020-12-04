import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.elliptical_coil

import unittest


class TestSimpleEllipticalTFSet(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.elliptical_coil.SimpleEllipticalTFSet()

        prob.setup(force_alloc_complex=True)

        prob.set_val('elongation_multiplier', 0.7)
        prob.set_val('κ', 2.73977961)
        prob.set_val('R0', 3)
        prob.set_val('r2', 8.168)
        prob.set_val('r1', 0.26127)
        prob.set_val('cross section', 0.052)
        prob.set_val('n_coil', 18)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

        prob.run_driver()


if __name__ == '__main__':
    unittest.main()
