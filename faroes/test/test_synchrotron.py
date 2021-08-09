import faroes.synchrotron as sy
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_near_equal
import unittest


class TestSynchrotronFit(unittest.TestCase):
    r"""
    Tests if derivatives of synchrotron fit are equivalent under
    analytical and numerical calculations.
    """

    def setUp(self):
        prob = om.Problem()
        prob.model = sy.SynchrotronFit()

        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("a0", 2)
        prob.set_val("κ", 1)

        prob.set_val("αn", 2)
        prob.set_val("αT", 3)
        prob.set_val("βT", 2)

        prob.set_val("Bt", 4)
        prob.set_val("pa", 3)
        prob.set_val("r", 0.5)
        prob.set_val("n0", 2)
        prob.set_val("T0", 5)

        self.prob = prob

    def test_partials(self):
        prob = self.prob
        check = prob.check_partials(out_stream=None, method='cs', step=1e-10)
        assert_check_partials(check)

        prob.run_driver()


class TestSynchrotronWithTriangularity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = sy.Synchrotron(implement_triangularity=True)
        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("a0", 2)
        prob.set_val("κ", 1)
        prob.set_val("δ", 0.3)
        prob.set_val("αn", 2)
        prob.set_val("αT", 3)
        prob.set_val("βT", 2)
        prob.set_val("Bt", 5)
        prob.set_val("r", 0.5)
        prob.set_val("n0", 2)
        prob.set_val("T0", 5)
        self.prob = prob

    def test_value_triangularity(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val("P"), 0.11649944, tolerance=1e-4)


class TestSynchrotronWithoutTriangularity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        prob.model = sy.Synchrotron(implement_triangularity=False)
        prob.setup(force_alloc_complex=True)
        prob.set_val("A", 5 / 2)
        prob.set_val("a0", 2)
        prob.set_val("κ", 1)
        prob.set_val("δ", 0.3)
        prob.set_val("αn", 2)
        prob.set_val("αT", 3)
        prob.set_val("βT", 2)
        prob.set_val("Bt", 5)
        prob.set_val("r", 0.5)
        prob.set_val("n0", 2)
        prob.set_val("T0", 5)
        self.prob = prob

    def test_value_no_triangularity(self):
        prob = self.prob
        prob.run_driver()
        assert_near_equal(prob.get_val("P"), 0.12081842, tolerance=1e-4)


if __name__ == '__main__':
    unittest.main()
