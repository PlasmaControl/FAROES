import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import faroes.elliptical_plasma

import unittest
import numpy.random as nprand


class TestPlasmaGeometry(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.elliptical_plasma.PlasmaGeometry()
        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
