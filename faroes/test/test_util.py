from faroes.configurator import UserConfigurator
import faroes.util as util

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import numpy as np

import unittest


class OffsetParametricCurvePoints(unittest.TestCase):
    def setUp(self):
        x = [1, 1, 0, 0]
        y = [0, 1, 1, 0]
        dx_dt = [1, -1, -1, 1]
        dy_dt = [1, 1, -1, -1]

        opc = util.OffsetParametricCurvePoints()
        prob = om.Problem()
        ivc = om.IndepVarComp()
        ivc.add_output("x", val=x, units="m")
        ivc.add_output("dx_dt", val=dx_dt, units="m")
        ivc.add_output("y", val=y, units="m")
        ivc.add_output("dy_dt", val=dy_dt, units="m")
        prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.model.add_subsystem("opc", opc, promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("s", 2**(1 / 2), units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()


if __name__ == "__main__":
    unittest.main()
