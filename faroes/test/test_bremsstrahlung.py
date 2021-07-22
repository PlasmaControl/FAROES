import faroes.bremsstrahlung as br
import openmdao.api as om
import unittest
from openmdao.utils.assert_utils import assert_near_equal


class TestPedestalEqualsParabolic(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = br.Bremsstrahlung(profile="pedestal",
                                       triangularity="constant")
        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("δ0", 0.2)
        prob.set_val("κ", 1)
        prob.set_val("a0", 5)
        prob.set_val("Zeff", 4)
        prob.set_val("n0", 2e20)
        prob.set_val("T0", 2, units="keV")
        prob.set_val("αn", 0.8)
        prob.set_val("αT", 0.8)
        prob.set_val("β", 2)
        prob.set_val("ρpedn", 1)
        prob.set_val("ρpedT", 1)
        prob.set_val("nped", 0)
        prob.set_val("n1", 0)
        prob.set_val("Tped", 0)
        prob.set_val("T1", 0)
        self.prob = prob

        prob1 = om.Problem()
        prob1.model = br.Bremsstrahlung(profile="parabolic",
                                        triangularity="constant")
        prob1.setup(force_alloc_complex=True)
        prob1.set_val("A", 5 / 2)
        prob1.set_val("δ0", 0.2)
        prob1.set_val("κ", 1)
        prob1.set_val("α", 2)
        prob1.set_val("a0", 5)
        prob1.set_val("Zeff", 4)
        prob1.set_val("n0", 2e20)
        prob1.set_val("T0", 2, units="keV")
        self.prob1 = prob1

        prob2 = om.Problem()
        prob2.model = br.Bremsstrahlung(profile="pedestal",
                                        triangularity="linear")
        prob2.setup(force_alloc_complex=True)
        prob2.set_val("A", 5 / 2)
        prob2.set_val("δ0", 0.2)
        prob2.set_val("κ", 1)
        prob2.set_val("a0", 5)
        prob2.set_val("Zeff", 4)
        prob2.set_val("n0", 2e20)
        prob2.set_val("T0", 2, units="keV")
        prob2.set_val("αn", 0.8)
        prob2.set_val("αT", 0.8)
        prob2.set_val("β", 2)
        prob2.set_val("ρpedn", 1)
        prob2.set_val("ρpedT", 1)
        prob2.set_val("nped", 0)
        prob2.set_val("n1", 0)
        prob2.set_val("Tped", 0)
        prob2.set_val("T1", 0)
        self.prob2 = prob2

        prob3 = om.Problem()
        prob3.model = br.Bremsstrahlung(profile="parabolic",
                                        triangularity="linear")
        prob3.setup(force_alloc_complex=True)
        prob3.set_val("A", 5 / 2)
        prob3.set_val("δ0", 0.2)
        prob3.set_val("κ", 1)
        prob3.set_val("α", 2)
        prob3.set_val("a0", 5)
        prob3.set_val("Zeff", 4)
        prob3.set_val("n0", 2e20)
        prob3.set_val("T0", 2, units="keV")
        self.prob3 = prob3

    def test_is_parabolic_constant(self):
        prob = self.prob
        prob.run_driver()
        prob1 = self.prob1
        prob1.run_driver()
        assert_near_equal(prob.get_val("P"), prob1.get_val("P"),
                          tolerance=1e-2)

    def test_is_parabolic_linear(self):
        prob2 = self.prob2
        prob2.run_driver()
        prob3 = self.prob3
        prob3.run_driver()
        assert_near_equal(prob2.get_val("P"), prob3.get_val("P"),
                          tolerance=1e-2)


class TestPedestal(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = br.Bremsstrahlung(profile="pedestal",
                                       triangularity="constant")
        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("δ0", 0.2)
        prob.set_val("κ", 1)
        prob.set_val("a0", 5)
        prob.set_val("Zeff", 4)
        prob.set_val("n0", 2e20)
        prob.set_val("T0", 2, units="keV")
        prob.set_val("αn", 0.8)
        prob.set_val("αT", 0.8)
        prob.set_val("β", 1.5)
        prob.set_val("ρpedn", 0.5)
        prob.set_val("ρpedT", 0.5)
        prob.set_val("nped", 3e20)
        prob.set_val("n1", 4e20)
        prob.set_val("Tped", 2.5, units="keV")
        prob.set_val("T1", 3.5, units="keV")
        self.prob = prob

        prob1 = om.Problem()
        prob1.model = br.Bremsstrahlung(profile="pedestal",
                                        triangularity="linear")
        prob1.setup(force_alloc_complex=True)
        prob1.set_val("A", 5 / 2)
        prob1.set_val("δ0", 0.2)
        prob1.set_val("κ", 1)
        prob1.set_val("a0", 5)
        prob1.set_val("Zeff", 4)
        prob1.set_val("n0", 2e20)
        prob1.set_val("T0", 2, units="keV")
        prob1.set_val("αn", 0.8)
        prob1.set_val("αT", 0.8)
        prob1.set_val("β", 1.5)
        prob1.set_val("ρpedn", 0.5)
        prob1.set_val("ρpedT", 0.5)
        prob1.set_val("nped", 3e20)
        prob1.set_val("n1", 4e20)
        prob1.set_val("Tped", 2.5, units="keV")
        prob1.set_val("T1", 3.5, units="keV")
        self.prob1 = prob1

    def test_value_constant(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val("P"), 9.80328159e+09, tolerance=1e-4)

    def test_value_linear(self):
        prob1 = self.prob1
        prob1.run_driver()
        assert_near_equal(prob1.get_val("P"), 9.77669231e+09, tolerance=1e-4)


if __name__ == '__main__':
    unittest.main()
