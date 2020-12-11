import faroes.blanket
from faroes.configurator import UserConfigurator

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import unittest


class TestMenardSTBlanketAndShieldMagnetProtection(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()

        prob.model = faroes.blanket.MenardSTBlanketAndShieldMagnetProtection()

        prob.setup(force_alloc_complex=True)

        prob.set_val('Ib blanket thickness', 0.1)
        prob.set_val('Ib WC shield thickness', 0.5)
        prob.set_val('Ib WC VV shield thickness', 0.1)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_loading(self):
        prob = om.Problem()
        uc = UserConfigurator()
        prob.model = faroes.blanket.MenardSTBlanketAndShieldMagnetProtection(
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
        prob.model = faroes.blanket.MenardSTBlanketAndShieldGeometry()

        prob.setup(force_alloc_complex=True)

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
