import faroes.generomakcosting as gc

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import unittest


class TestFusionIslandCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.FusionIslandCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_t", 2000, units="MW")
        prob.set_val("Cpc", 500, units="MUSD")
        prob.set_val("Csg", 500, units="MUSD")
        prob.set_val("Cst", 500, units="MUSD")
        prob.set_val("C_aux", 500, units="MUSD")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestAuxHeatingCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.AuxHeatingCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_aux", 50, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestPrimaryCoilSetCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.PrimaryCoilSetCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("V_pc", 30, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestStructureCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.StructureCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("V_st", 30, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestGeneromakStructureVolume(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.GeneromakStructureVolume()

        prob.setup(force_alloc_complex=True)
        prob.set_val("V_pc", 1000, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestBlanketCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.BlanketCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("V_bl", 40, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestShieldWithGapsCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.ShieldWithGapsCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("V_sg", 40, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestDeuteriumCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.DeuteriumCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_fus", 1500, units="MW")
        prob.set_val("f_av", 0.8, units=None)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestCapitalCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.CapitalCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_t", 2000, units="MW")
        prob.set_val("P_e", 1000, units="MW")
        prob.set_val("C_FI", 500, units="MUSD")
        prob.set_val("V_FI", 5000, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestFuelCycleCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.FuelCycleCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("C_ba", 100, units="MUSD/a")
        prob.set_val("C_ta", 200, units="MUSD/a")
        prob.set_val("C_aa", 300, units="MUSD/a")
        prob.set_val("C_fa", 400, units="MUSD/a")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestAveragedAnnualBlanketCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.AveragedAnnualBlanketCost()

        prob.setup(force_alloc_complex=True)

        prob.set_val("C_bl", 20, units="MUSD")
        prob.set_val("f_av", 0.9)
        prob.set_val("F_wn", 10, units="MW*a/m**2")
        prob.set_val("p_wn", 10, units="MW/m**2")
        prob.set_val("N_years", 40)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestAveragedAnnualDivertorCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.AveragedAnnualDivertorCost()

        prob.setup(force_alloc_complex=True)

        prob.set_val("C_tt", 20, units="MUSD")
        prob.set_val("f_av", 0.9)
        prob.set_val("F_tt", 10, units="MW*a/m**2")
        prob.set_val("p_tt", 10, units="MW/m**2")
        prob.set_val("N_years", 40)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestCostOfElectricity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.CostOfElectricity()

        prob.setup(force_alloc_complex=True)

        prob.set_val("f_av", 0.9)
        prob.set_val("C_CO", 3000, units="MUSD")
        prob.set_val("C_F", 3000, units="MUSD/a")
        prob.set_val("C_OM", 300, units="MUSD/a")
        prob.set_val("P_e", 500, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestFixedOMCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.FixedOMCost()

        prob.setup(force_alloc_complex=True)

        prob.set_val("P_e", 500, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestMiscReplacements(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.MiscReplacements()

        prob.setup(force_alloc_complex=True)

        prob.set_val("C_fuel", 0.9, units="MUSD/a")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestTotalCapitalCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.TotalCapitalCost()

        prob.setup(force_alloc_complex=True)

        prob.set_val("C_D", 500, units="MUSD")
        prob.set_val("f_CAPO", 1.063)
        prob.set_val("f_IND", 1.063)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class IndirectChargesFactor(om.ExplicitComponent):
    r"""Owner's cost proportional to construction time

    Equation (28) of [1]_.

    Inputs
    ------
    T_constr : float
        a, Construction time

    Outputs
    -------
    f_IND : float
        Indirect charges factor
    """


if __name__ == '__main__':
    unittest.main()
