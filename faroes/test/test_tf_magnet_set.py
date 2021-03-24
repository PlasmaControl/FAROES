import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.tf_magnet_set as tfset

import unittest


class TestTFMagnetSet(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = tfset.TFMagnetSet()

        prob.setup(force_alloc_complex=True)
        prob.set_val('I_leg', 5.1, 'MA')
        prob.set_val('arc length', 20, 'm')
        prob.set_val('cross section', 0.8, 'm**2')
        prob.set_val('n_coil', 18)

        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == "__main__":
    unittest.main()
