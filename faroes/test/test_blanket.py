import faroes.blanket as blanket
from faroes.configurator import UserConfigurator

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal

import unittest
from importlib import resources


class TestMenardInboardBlanketFitDoubleReLu(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "blanket_thickness_1.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = blanket.MenardInboardBlanketFit(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 2.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        th = prob.get_val("blanket_thickness", units="m")
        expected = 0.19597
        assert_near_equal(th, expected, tolerance=1e-4)


class TestOutboardBlanketFitDoubleReLu(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "blanket_thickness_1.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = blanket.OutboardBlanketFit(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 4.0)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        th = prob.get_val("blanket_thickness", units="m")
        expected = 0.76
        assert_near_equal(th, expected, tolerance=1e-4)


class TestMenardInboardShieldFitDoubleReLu(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "blanket_thickness_1.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = blanket.MenardInboardShieldFit(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 2.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        th = prob.get_val("shield_thickness", units="m")
        expected = 0.402015
        assert_near_equal(th, expected, tolerance=1e-4)


class TestMenardInboardBlanketFitConstant(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "blanket_thickness_2.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = blanket.MenardInboardBlanketFit(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 2.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        th = prob.get_val("blanket_thickness", units="m")
        expected = 0.0
        assert_near_equal(th, expected, tolerance=1e-4)


class TestMenardInboardShieldFitConstant(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        resource_dir = "faroes.test.test_data"
        resource_name = "blanket_thickness_2.yaml"
        if resources.is_resource(resource_dir, resource_name):
            with resources.path(resource_dir, resource_name) as path:
                uc = UserConfigurator(user_data_file=path)

        prob.model = blanket.MenardInboardShieldFit(config=uc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 2.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob
        prob.run_driver()
        th = prob.get_val("shield_thickness", units="m")
        expected = 0.5
        assert_near_equal(th, expected, tolerance=1e-4)


class TestMenardSTBlanketAndShieldMagnetProtection(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = blanket.MenardSTBlanketAndShieldMagnetProtection()

        prob.setup(force_alloc_complex=True)

        prob.set_val('Ib blanket thickness', 0.1)
        prob.set_val('Ib WC shield thickness', 0.5)
        prob.set_val('Ib WC VV shield thickness', 0.1)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_loading(self):
        prob = om.Problem()
        uc = UserConfigurator()
        prob.model = blanket.MenardSTBlanketAndShieldMagnetProtection(
            config=uc)

        prob.setup(force_alloc_complex=True)

        prob.set_val('Ib blanket thickness', 0.1)
        prob.set_val('Ib WC shield thickness', 0.5)
        prob.set_val('Ib WC VV shield thickness', 0.1)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMenardSTBlanketAndShieldGeometry(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model = blanket.MenardSTBlanketAndShieldGeometry()

        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestInboardMidplaneNeutronFluxFromRing(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = blanket.InboardMidplaneNeutronFluxFromRing()
        prob.setup(force_alloc_complex=True)

        prob.set_val('R0', 3.0)
        prob.set_val('r_in', 2.0)

        prob.set_val('P_n', 100, units="MW")
        prob.set_val('S', 1e20, units="s**-1")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        q_n = prob.get_val("q_n", units="MW/m**2")
        expected = 0.688
        assert_near_equal(q_n, expected, tolerance=1e-2)


class TestMenardMagnetCooling(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = blanket.MenardMagnetCooling()
        prob.setup(force_alloc_complex=True)

        prob.set_val('P_n', 278)
        prob.set_val('Δr_sh', 0.6)
        prob.set_val('f_refrigeration', 20)

        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        P_c = prob.get_val("P_c,el", units="MW")
        expected = 0.2024
        assert_near_equal(P_c, expected, tolerance=1e-3)
        sh = prob.get_val("shielding")
        expected = 0.0000364
        assert_near_equal(sh, expected, tolerance=1e-3)


class TestRefrigerationPerformance(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = blanket.RefrigerationPerformance()
        prob.setup(force_alloc_complex=True)

        prob.set_val("T_cold", 40, units="K")
        prob.set_val("T_hot", 300, units="K")
        prob.set_val("FOM", 0.325)

        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        f = prob.get_val("f")
        expected = 20
        assert_near_equal(f, expected, tolerance=1e-3)


class TestMenardMagnetLifetime(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = blanket.MenardMagnetLifetime()
        prob.setup(force_alloc_complex=True)

        prob.set_val("Shielding factor", 1.5615)
        prob.set_val("q_n_IB", 0.8860, units="MW/m**2")
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_value(self):
        prob = self.prob
        prob.run_driver()
        lifetime = prob.get_val("lifetime")
        expected = 5.773
        assert_near_equal(lifetime, expected, tolerance=1e-3)

    def test_loading(self):
        prob = om.Problem()
        uc = UserConfigurator()
        prob.model = blanket.MenardMagnetLifetime(config=uc)


if __name__ == '__main__':
    unittest.main()
