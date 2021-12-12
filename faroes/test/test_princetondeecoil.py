import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import faroes.princetondeecoil as coil
import numpy as np
from scipy.constants import pi

import unittest


class TestPrincetonDeeTFSet(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        θ = np.linspace(-pi, pi, 31, endpoint=True)

        prob.model.add_subsystem("ivc",
                                 om.IndepVarComp("θ", val=θ),
                                 promotes_outputs=["*"])

        prob.model.add_subsystem("tadTF",
                                 coil.PrincetonDeeTFSet(),
                                 promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("R0", 2)
        prob.set_val("Ib TF R_out", 1)
        prob.set_val("Ob TF R_in", 3)
        prob.set_val("θ", θ)

        self.prob = prob

    def test_partials(self):
        prob = self.prob
        # Several of the calculations are done exactly,
        # but some have fd derivatives, so we need to put
        # form="central" here for the check.
        check = prob.check_partials(out_stream=None,
                                    method="fd",
                                    form="central")
        assert_check_partials(check, rtol=2e-5, atol=8e-5)

    def test_values(self):
        prob = self.prob

        prob.set_val("R0", 1)
        prob.set_val("Ib TF R_out", np.exp(-1))
        prob.set_val("Ob TF R_in", np.exp(1))

        prob.run_driver()

        expected = 76.5266
        V_enc = prob.get_val("tadTF.V_enc", units="m**3")
        assert_near_equal(V_enc, expected, tolerance=1e-6)

    def test_half_height(self):
        prob = self.prob

        prob.set_val("R0", 1)
        prob.set_val("Ib TF R_out", np.exp(-1))
        prob.set_val("Ob TF R_in", np.exp(1))

        prob.run_driver()

        expected = 2.24395
        hh = prob.get_val("tadTF.half-height", units="m")
        assert_near_equal(hh, expected, tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()
