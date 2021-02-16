from faroes.configurator import UserConfigurator
import faroes.sauter_plasma as plasma

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import numpy as np

import unittest


class TestSauterGeometry(unittest.TestCase):
    def setUp(self):
        θ = np.linspace(0, 2 * np.pi, 7, endpoint=False)

        uc = UserConfigurator()
        sg = plasma.SauterGeometry(config=uc)
        prob = om.Problem()

        prob.model.add_subsystem("ivc",
                                 om.IndepVarComp("θ", val=θ),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("geom", sg, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val('R0', 3, 'm')
        prob.set_val('A', 1.6)
        prob.set_val('κ', 2.7)
        prob.set_val('δ', 0.5)
        prob.set_val('ξ', 0.3)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
