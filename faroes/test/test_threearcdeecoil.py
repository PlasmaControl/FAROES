import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.threearcdeecoil as coil
import numpy as np
from scipy.constants import pi

import unittest


class TestThreeArcDeeTFSet(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        θ = np.linspace(-pi, pi, 31, endpoint=True)

        prob.model.add_subsystem("ivc",
                                 om.IndepVarComp("θ", val=θ),
                                 promotes_outputs=["*"])

        prob.model.add_subsystem("tadTF",
                                 coil.ThreeArcDeeTFSet(),
                                 promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("R0", 3)
        prob.set_val("Ib TF R_out", 0.94)
        prob.set_val("r_c", 1.2)
        prob.set_val("e_a", 3)
        prob.set_val("hhs", 3)
        prob.set_val("θ", θ)

        check = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(check)

        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(check)


if __name__ == "__main__":
    unittest.main()