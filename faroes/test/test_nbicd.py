from faroes.configurator import UserConfigurator
import faroes.nbicd

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

import numpy as np

import unittest


class TestCurrentDriveBeta1(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.CurrentDriveBeta1()

        prob.setup(force_alloc_complex=True)
        prob.set_val('Ab', 2)
        prob.set_val('Ai', 2.5)
        prob.set_val('Z_eff', 2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["β1"], 2.5, tolerance=1e-3)


class TestCurrentDriveAlphaCubed(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model.add_subsystem('ivc',
                                 om.IndepVarComp('ni',
                                                 val=np.ones(3),
                                                 units='n20'),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem('cda',
                                 faroes.nbicd.CurrentDriveAlphaCubed(),
                                 promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("ni", np.array([0.4240, 0.424, 0.035333]), units="n20")
        prob.set_val("Ai", [2, 3, 12], units="u")
        prob.set_val("Zi", [1, 1, 6])

        prob.set_val("ne", 1.06e20, units="m**-3")
        prob.set_val("ve", 56922, units="km/s")
        prob.set_val("v0", 6922, units="km/s")
        self.prob = prob

    def test_value(self):
        prob = self.prob
        expected_α3 = 0.1757
        prob.run_driver()
        assert_near_equal(prob["cda.α³"], expected_α3, tolerance=1e-2)

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestCurrentDriveA(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.CurrentDriveA()

        prob.setup(force_alloc_complex=True)
        prob.set_val('vb', 10)
        prob.set_val('vth_e', 60)
        prob.set_val('Z_eff', 1.2)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["A"], 10 / 7, tolerance=1e-3)


class TestTrappedParticleFractionUpperEst(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.TrappedParticleFractionUpperEst()

        prob.setup(force_alloc_complex=True)
        prob.set_val('ε', 0.1)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)

    def test_values(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["ftrap_u"], 0.455802, tolerance=1e-4)


class TestCurrentDriveEfficiencyTerms(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = faroes.nbicd.CurrentDriveEfficiencyTerms()

        prob.setup(force_alloc_complex=True)
        prob.set_val('Zb', 1)
        prob.set_val("v0", 6922, units='km/s')
        prob.set_val("E_NBI", 500, units='keV')
        prob.set_val("G", 0.97)
        prob.set_val('α³', 0.32)
        prob.set_val('τs', 0.599, units="s")
        prob.set_val('R', 3.0, units="m")
        prob.set_val("<T_e>", 9.20, units="keV")
        prob.set_val("β1", 2.50)
        self.prob = prob

    def test_partials(self):
        prob = self.prob
        prob.run_model()
        check = prob.check_partials(out_stream=None,
                                    method='fd',
                                    form='central')
        assert_check_partials(check, rtol=1e-5)

    def test_values1(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["term1"], 0.3233, tolerance=1e-3)

    def test_values2(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["term2"], 1.007, tolerance=1e-3)

    def test_values3(self):
        prob = self.prob

        prob.run_driver()
        assert_near_equal(prob["term3"], 0.307878, tolerance=1e-3)


class TestCurrentDriveEfficiency(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        uc = UserConfigurator()

        prob.model.add_subsystem("ivc",
                                 om.IndepVarComp("ni",
                                                 val=np.ones(3),
                                                 units="n20"),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem(
            "cde",
            faroes.nbicd.CurrentDriveEfficiency(config=uc),
            promotes_inputs=["*"])

        prob.setup()

        prob.set_val("Ab", 2, units="u")
        prob.set_val("Zb", 1)
        prob.set_val("vb", 6922, units="km/s")
        prob.set_val("Eb", 500, units="keV")

        prob.set_val("R0", 3.0, units="m")
        prob.set_val("ε", 1 / 1.6)

        prob.set_val("Z_eff", 2)
        prob.set_val("ne", 1.06, units="n20")
        prob.set_val("<T_e>", 9.20, units="keV")
        prob.set_val("vth_e", 56922, units="km/s")

        prob.set_val("τs", 0.599, units="s")

        prob.set_val("ni", np.array([0.424, 0.424, 0.0353]), units="n20")
        prob.set_val("Ai", [2, 3, 12], units="u")
        prob.set_val("Zi", [1, 1, 6])
        self.prob = prob

    def test_val(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob["cde.It/P"], 0.1313, tolerance=1e-3)


class TestNBICurrent(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model.add_subsystem('ivc',
                                 om.IndepVarComp('S',
                                                 val=np.ones(3),
                                                 units='1/s'),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem('cd',
                                 faroes.nbicd.NBICurrent(),
                                 promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val("S", np.array([7.0e20, 4.0e20, 2.0e20]), units="1/s")
        prob.set_val("It/P", [0.1, 0.1, 0.1])
        prob.set_val("Eb", [500, 250, 166], units="keV")

        self.prob = prob

    def test_value(self):
        prob = self.prob
        expected_I_NBI = 7.7417
        prob.run_driver()
        assert_near_equal(prob["cd.I_NBI"], expected_I_NBI, tolerance=1e-3)

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None,
                                    method='cs',
                                    excludes=["S"])
        check = prob.check_partials(out_stream=None,
                                    method='cs',
                                    includes=["S"],
                                    step=1e18j)
        assert_check_partials(check)


if __name__ == '__main__':
    unittest.main()
