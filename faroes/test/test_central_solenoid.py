import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import faroes.central_solenoid

import unittest


class TestThinSolenoidInductance(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.central_solenoid.ThinSolenoidInductance()

        prob.setup()

        check = prob.check_partials(out_stream=None, method='fd')
        assert_check_partials(check, rtol=5e-5)


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
