import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.cryostat

import unittest


class TestCryostat(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.cryostat.SimpleCryostat()

        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
