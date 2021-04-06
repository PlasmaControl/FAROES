import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.ripple as rp

import unittest


class TestCryostat(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = rp.SimpleRipple()
        prob.setup(force_alloc_complex=True)
        prob.set_val("R", 5)
        prob.set_val("r2", 8.168)
        prob.set_val("r1", 0.26127)
        prob.set_val("n_coil", 18)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
