import faroes.sol as sol
from faroes.configurator import UserConfigurator

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import unittest
from importlib import resources


class TestGoldstonHDSOL(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = sol.GoldstonHDSOL()

        prob.setup(force_alloc_complex=True)
        prob.set_val("R0", 3.0, units="m")
        prob.set_val("a", 1.875, units="m")
        prob.set_val("κ", 2.74)
        prob.set_val("Bt", 2.094, units="T")
        prob.set_val("Ip", 14.67, units="MA")
        prob.set_val("P_sol", 23.81, units="MW")
        prob.set_val("Z_eff", 2)
        prob.set_val("Z-bar", 1)
        prob.set_val("A-bar", 2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        expected = 96.527
        T_sep = prob.get_val("T_sep", units="eV")
        assert_near_equal(T_sep, expected, tolerance=1e-4)

        expected = 0.00331
        λ = prob.get_val("λ", units="m")
        assert_near_equal(λ, expected, tolerance=1e-3)


class TestStrikePointRadius1(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "sol_options_1.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = sol.StrikePointRadius(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("R0", 3.0, units="m")
        prob.set_val("a", 1.875, units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        Rs = prob.get_val("R_strike", units="m")
        expected = 2.4375
        assert_near_equal(Rs, expected, tolerance=1e-4)


class TestStrikePointRadius2(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "sol_options_2.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = sol.StrikePointRadius(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("R0", 3.0, units="m")
        prob.set_val("a", 1.875, units="m")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        Rs = prob.get_val("R_strike", units="m")
        expected = 4.8
        assert_near_equal(Rs, expected, tolerance=1e-4)


class TestStrikePointRadius3(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "sol_options_3.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = sol.StrikePointRadius(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("R0", 3.0, units="m")
        prob.set_val("a", 1.875, units="m")
        prob.set_val("δ", 0.5)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        Rs = prob.get_val("R_strike", units="m")
        expected = 2.53125
        assert_near_equal(Rs, expected, tolerance=1e-4)


class TestPeakHeatFlux1(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "sol_options_1.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = sol.PeakHeatFlux(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("κ", 2.74)
        prob.set_val("R_strike", 2.44, units="m")
        prob.set_val("P_sol", 23.81, units="MW")
        prob.set_val("q_star", 3.56)
        prob.set_val("λ_sol", 3.31, units="mm")
        prob.set_val("f_outer", 0.8)
        prob.set_val("f_fluxexp", 22)
        prob.set_val("θ_pol", 23, units="deg")
        prob.set_val("θ_tot", 1, units="deg")
        prob.set_val("N_div", 2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        q_max = prob.get_val("q_max", units="MW/m**2")
        expected = 3.339
        assert_near_equal(q_max, expected, tolerance=2e-3)


class TestPeakHeatFlux2(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "sol_options_2.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = sol.PeakHeatFlux(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("κ", 2.74)
        prob.set_val("R_strike", 2.44, units="m")
        prob.set_val("P_sol", 23.81, units="MW")
        prob.set_val("q_star", 3.56)
        prob.set_val("λ_sol", 3.31, units="mm")
        prob.set_val("f_outer", 0.8)
        prob.set_val("f_fluxexp", 22)
        prob.set_val("θ_pol", 23, units="deg")
        prob.set_val("θ_tot", 1, units="deg")
        prob.set_val("N_div", 2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        q_max = prob.get_val("q_max", units="MW/m**2")
        expected = 4.25
        assert_near_equal(q_max, expected, tolerance=2e-3)


if __name__ == '__main__':
    unittest.main()
