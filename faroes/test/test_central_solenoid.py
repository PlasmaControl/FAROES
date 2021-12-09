import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import faroes.central_solenoid

import unittest


class TestThinSolenoidInductance(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.central_solenoid.ThinSolenoidInductance()

        prob.setup()
        prob.set_val('r', 1.305118)
        prob.set_val('h', 5, 'm')
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        prob.run_driver()

        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, rtol=8e-5)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        expected = 2.72822e-5
        L_line = prob.get_val("L_line", units="H*m**2")
        assert_near_equal(L_line, expected, tolerance=1e-4)


class TestFiniteBuildCentralSolenoid(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.central_solenoid.FiniteBuildCentralSolenoid()

        prob.setup(force_alloc_complex=True)

        prob.set_val('R_in', 1.0)
        prob.set_val('R_out', 1.5)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestThinSolenoidStoredEnergy(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.central_solenoid.ThinSolenoidStoredEnergy()

        prob.setup(force_alloc_complex=True)

        prob.set_val('jl', 12)
        prob.set_val('L_line', 2.72e-5)

        prob.run_driver()

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestFiniteSolenoidStresses(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.central_solenoid.FiniteSolenoidStresses()

        prob.setup(force_alloc_complex=True)

        prob.set_val('R_in', 1.0)
        prob.set_val('R_out', 1.5)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
