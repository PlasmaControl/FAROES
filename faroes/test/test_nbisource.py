import faroes.nbisource
from faroes.configurator import UserConfigurator
from plasmapy.particles import deuteron
from openmdao.utils.assert_utils import assert_near_equal

import openmdao.api as om

import unittest


class TestNBISource(unittest.TestCase):
    def test_vals(self):
        prob = om.Problem()
        uc = UserConfigurator()

        prob.model = faroes.nbisource.SimpleNBISource(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val('E', 500, units='keV')
        prob.set_val('P', 50, units='MW')
        prob.set_val('m', deuteron.mass, units='kg')

        prob.run_driver()
        assert_near_equal(prob["f"], 6.2415094e20, tolerance=1e-3)
        assert_near_equal(prob["v"], 6.92e6, tolerance=1e-3)



if __name__ == '__main__':
    unittest.main()
