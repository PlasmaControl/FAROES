import faroes.generomakcosting as gc

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import unittest


class TestFusionIslandCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        fi_cc = {
            'c_Pt': 0.221,
            'd_Pt': 4150,
            'e_Pt': 0.6,
            'm_pc': 1.5,
            'm_sg': 1.25,
            'm_st': 1.0,
            'm_aux': 1.1,
            'fudge': 1.0,
        }
        prob.model = gc.FusionIslandCost(cost_params=fi_cc)
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

        prob.model = gc.AuxHeatingCost(cost_per_watt=5.3)

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

        cost_per_vol = 1.66  # MUSD / m^3
        prob.model = gc.PrimaryCoilSetCost(cost_per_volume=cost_per_vol)

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

        cost_per_vol = 0.36  # MUSD / m^3
        prob.model = gc.StructureCost(cost_per_volume=cost_per_vol)

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

        cost_per_vol = 0.75  # MUSD / m^3
        prob.model = gc.BlanketCost(cost_per_volume=cost_per_vol)

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

        cost_per_vol = 0.29  # MUSD / m^3
        prob.model = gc.ShieldWithGapsCost(cost_per_volume=cost_per_vol)

        prob.setup(force_alloc_complex=True)
        prob.set_val("V_sg", 40, units="m**3")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestDeuteriumVariableCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        deu_cc = {"C_deu_per_kg": 10000.0}
        prob.model = gc.DeuteriumVariableCost(cost_params=deu_cc)

        prob.setup(force_alloc_complex=True)
        prob.set_val("P_fus", 1500, units="MW")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestAnnualDeuteriumCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.AnnualDeuteriumCost()

        prob.setup(force_alloc_complex=True)
        prob.set_val("C_Dv", 60, units="USD/h")
        prob.set_val("D usage", 6, units="g/h")
        prob.set_val("f_av", 0.8, units=None)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)
        prob.run_driver()


class TestCapitalCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        cap_cc = {
            'f_cont': 1.15,
            'c_e1': 0.9,
            'c_e2': 0.9,
            'c_e3': 1200,
            'd_Pt': 4150,
            'e_Pt': 0.6,
            'c_V': 0.839,
            'd_V': 5100,
            'e_V': 0.67,
            'fudge': 1.0,
        }
        prob.model = gc.CapitalCost(cost_params=cap_cc)
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
        ann_bl_cc = {
            'f_failures': 1.1,
            'f_spares': 1.1,
            'F_CR0': 0.078,
            'fudge': 1.0,
        }

        prob.model = gc.AveragedAnnualBlanketCost(cost_params=ann_bl_cc)

        prob.setup(force_alloc_complex=True)

        prob.set_val("C_bl", 20, units="MUSD")
        prob.set_val("p_wn", 5, units="MW/m**2")
        prob.set_val("F_wn", 15, units="MW*a/m**2")
        prob.set_val("f_av", 0.9)
        prob.set_val("N_years", 40)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestAveragedAnnualDivertorCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        ann_dv_cc = {
            'f_failures': 1.2,
            'f_spares': 1.1,
            'F_CR0': 0.078,
            'fudge': 1.0,
        }

        prob.model = gc.AveragedAnnualDivertorCost(cost_params=ann_dv_cc)

        prob.setup(force_alloc_complex=True)

        prob.set_val("C_tt", 20, units="MUSD")
        prob.set_val("F_tt", 20, units="MW*a/m**2")
        prob.set_val("p_tt", 10, units="MW/m**2")
        prob.set_val("f_av", 0.9)
        prob.set_val("N_years", 40)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestCostOfElectricity(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        coe_cc = {
            'F_CR0': 0.078,
            'waste_charge': 0.5,
            'fudge': 1.0,
        }

        prob.model = gc.CostOfElectricity(cost_params=coe_cc)

        prob.setup(force_alloc_complex=True)

        prob.set_val("f_av", 0.9)
        prob.set_val("C_C0", 3000, units="MUSD")
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

        fom_cc = {
            "base_Pe": 1200,
            "base_OM": 108,
            "fudge": 1.0,
        }
        prob.model = gc.FixedOMCost(cost_params=fom_cc)

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

        misc_cc = {'f_CR0': 0.078, 'C_misc': 52.8}
        prob.model = gc.MiscReplacements(cost_params=misc_cc)

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
        prob.set_val("f_CAP0", 1.063)
        prob.set_val("f_IND", 1.063)
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestIndirectChargesFactor(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        prob.model = gc.IndirectChargesFactor()

        prob.setup(force_alloc_complex=True)

        prob.set_val("T_constr", 5, units="a")
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)


class TestGeneromakToGenX(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()
        cost_params = {
            'F_CR0': 0.078,
            'waste_charge': 0.5,
            'fudge': 1.0,
        }

        prob.model = gc.GeneromakToGenX(cost_params=cost_params)
        prob.setup(force_alloc_complex=True)
        prob.set_val('C_C0', 7, units='GUSD')
        prob.set_val('Initial blanket', 20, units='MUSD/a')
        prob.set_val('Initial divertor', 5, units='MUSD/a')

        prob.set_val('C_aa', 29.5, units='MUSD/a')
        prob.set_val('C_misca', 7.5, units='MUSD/a')
        prob.set_val('C_OM', 98.59, units='MUSD/a')

        prob.set_val('C_bv', 20, units='MUSD/a')
        prob.set_val('C_tv', 10, units='MUSD/a')
        prob.set_val('C_fv', 0.5, units='MUSD/a')

        prob.set_val('P_e', 1000, units='MW')
        self.prob = prob

    def test_partials(self):
        prob = self.prob

        check = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(check)
        prob.run_driver()


if __name__ == '__main__':
    unittest.main()
