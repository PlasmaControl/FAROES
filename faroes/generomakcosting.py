import openmdao.api as om

from scipy.constants import mega


class PrimaryCoilSetCost(om.ExplicitComponent):
    r"""Generomak primary TF coil set cost

    Inputs
    ------
    V_pc: float
       m**3, Material volume of primary (TF) coils

    Outputs
    -------
    Cpc: float
       MUSD, cost of the primary coilset

    Options
    -------
    cost_per_cubic_meter : float
        MUSD/m**3, Cost per cubic meter.

    Notes
    -----
    From the last column of Table III of [1]_.
    This uses the Adjusted for ITER Unit Cost values.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_pc", units="m**3", val=0.0)
        self.add_output("Cpc", units="MUSD", ref=100)

    def compute(self, inputs, outputs):
        outputs['Cpc'] = self.cpv * inputs["V_pc"]

    def setup_partials(self):
        self.declare_partials('Cpc', ['V_pc'], val=self.cpv)


class BlanketCost(om.ExplicitComponent):
    r"""Generomak blanket set cost

    Inputs
    ------
    V_bl: float
       m**3, Material volume of the blanket

    Outputs
    -------
    C_bl: float
       MUSD, cost of one blanket set

    Options
    -------
    cost_per_cubic_meter : float
        MUSD/m**3, Cost per cubic meter.

    Notes
    -----
    From the last column of Table III of [1]_.
    This uses the Adjusted for ITER Unit Cost values.

    In this model, the blanket is replaced, so it is considered as an operating
    cost rather than as a capital cost.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_bl", units="m**3")
        self.add_output("C_bl", units="MUSD", ref=100)

    def compute(self, inputs, outputs):
        outputs['C_bl'] = self.cpv * inputs["V_bl"]

    def setup_partials(self):
        self.declare_partials('C_bl', ['V_bl'], val=self.cpv)


class StructureCost(om.ExplicitComponent):
    r"""Generomak structure cost

    Inputs
    ------
    V_st: float
       m**3, Material volume of the inter-coil structures and gravity supports.

    Outputs
    -------
    Cst: float
       MUSD, cost of the inter-coil structures and gravity supports

    Options
    -------
    cost_per_cubic_meter : float
        MUSD/m**3, Cost per cubic meter.

    Notes
    -----
    From the last column of Table III of [1]_.
    This uses the Adjusted for ITER Unit Cost values.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_st", units="m**3")
        self.add_output("Cst", units="MUSD", ref=100)

    def compute(self, inputs, outputs):
        outputs['Cst'] = self.cpv * inputs["V_st"]

    def setup_partials(self):
        self.declare_partials('Cst', ['V_st'], val=self.cpv)


class ShieldWithGapsCost(om.ExplicitComponent):
    r"""Generomak shield with gaps cost

    Inputs
    ------
    V_sg: float
       m**3, Material volume of the shield (with gaps)

    Outputs
    -------
    Csg: float
       MUSD, cost of shielding (with gaps)

    Options
    -------
    cost_per_cubic_meter : float
        MUSD/m**3, Cost per cubic meter.

    Notes
    -----
    From the last column of Table III of [1]_.
    This uses the Adjusted for ITER Unit Cost values.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_sg", units="m**3")
        self.add_output("Csg", units="MUSD", ref=100)

    def compute(self, inputs, outputs):
        outputs['Csg'] = self.cpv * inputs["V_sg"]

    def setup_partials(self):
        self.declare_partials('Csg', ['V_sg'], val=self.cpv)


class DivertorCost(om.ExplicitComponent):
    r"""Generomak replaceable divertor cost

    Inputs
    ------
    A_tt: float
       m**2, Wall area taken up by divertor

    Outputs
    -------
    C_tt: float
       MUSD, cost of divertor

    Options
    -------
    cost_per_area : float
        MUSD/m**2, Cost per square meter.

    Notes
    -----
    From the paragraph below Equation (24) of [1]_.
    The Generomak estimates the target area as 10% of the wall area,
    and estimates a thermal load on the divertor targets of 10 MW/m^2
    when the neutron wall load p_wn is 3.0 MW/m^2. This doesn't seem compatible
    with modern ideas to have detached or very radiative divertors.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('cost_per_area')

    def setup(self):
        self.cpa = self.options['cost_per_area']
        self.add_input("A_tt", units="m**2")
        self.add_output("C_tt", units="MUSD", ref=3)

    def compute(self, inputs, outputs):
        outputs['C_tt'] = self.cpa * inputs["A_tt"]

    def setup_partials(self):
        self.declare_partials('C_tt', ['A_tt'], val=self.cpa)


class AuxHeatingCost(om.ExplicitComponent):
    r"""Generomak auxilliary heating cost

    Inputs
    ------
    P_aux: float
       MW, power of aux heating systems

    Outputs
    -------
    C_aux: float
       MUSD, cost of aux heating systems

    Options
    -------
    cost_per_watt : float
        USD/W, Cost of aux heating.

    Notes
    -----
    Dollar values are from the "Adjusted for ITER Unit Cost" column of Table
    III of [1]_.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('cost_per_watt')

    def setup(self):
        self.cpw = self.options['cost_per_watt']
        self.add_input("P_aux", units="MW")
        self.add_output("C_aux", units="MUSD", ref=100)

    def compute(self, inputs, outputs):
        outputs['C_aux'] = self.cpw * inputs["P_aux"]

    def setup_partials(self):
        self.declare_partials('C_aux', ['P_aux'], val=self.cpw)


class AnnualAuxHeatingCost(om.ExplicitComponent):
    r"""Part of the 'fuel cycle costs'

    Inputs
    ------
    C_aux: float
       MUSD, capital cost of aux heating

    Outputs
    -------
    C_aa:float
       MUSD/a, Averaged costs of aux heating

    Notes
    -----
    This is a constant factor of the aux heating capital cost.
    The value is from the paragraph before Equation (25) of [1]_.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('annual_aux_heating_costing', default=None)

    def setup(self):
        self.cc = self.options['annual_aux_heating_costing']
        self.add_input("C_aux", units="MUSD")
        self.add_output("C_aa", units="MUSD/a", ref=10)

    def compute(self, inputs, outputs):
        cc = self.cc
        fs = cc['f_spares']
        fa = cc['annual_aux_heating_factor']
        outputs['C_aa'] = fs * fa * inputs["C_aux"]

    def setup_partials(self):
        cc = self.cc
        fs = cc['f_spares']
        fa = cc['annual_aux_heating_factor']
        v = fs * fa
        self.declare_partials('C_aa', ['C_aux'], val=v)


class FusionIslandCost(om.ExplicitComponent):
    r"""Fusion island cost

    Equation (18) of [1]_.

    .. math::
         C_{FI} = c_{PT}(P_t / d_{Pt})^{e_{PT}}
                  + m_{pc} C_{pc}
                  + m_{sg} C_{sg}
                  + m_{st} C_{st}
                  + m_{aux} C_{aux}

    Inputs
    ------
    P_t : float
        MW, thermal power handled by the steam generators, etc

    Cpc : float
        GUSD, Cost of the primary (TF) coil set

    Csg : float
        GUSD, Cost of shielding-and-gaps

    Cst : float
        GUSD, Cost of the structure
    C_aux : float
        GUSD, Cost of auxilliary heating systems

    Outputs
    -------
    C_ht : float
        GUSD, cost of the "main heat transfer steam system". Note that the BOP
        such as turbine equipment is tabulated separately.
    C_FI: float
        GUSD, cost of the fusion island

    Options
    -------
    fusion_island_costing : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        c_Pt : float
            MUSD, Coefficient for the thermal conversion systems cost.
        d_Pt : float
            MW, Divisor for the thermal conversion systems cost.
        e_Pt : float
            Exponent for the thermal conversion systems cost.
        m_pc : float
            Multiplier for the cost of the primary coil set, Cpc,
            to account for the secondary coilsets and redundancy in the coil
            windings.
        m_sg : float
            Multiplier for the cost of shielding and gaps, Csg, to account
            for extra shielding around ports.
        m_st : float
            Multiplier for the structure cost Cst.
        m_aux : float
            Multiplier for the auxilliary heating cost C_aux, to allow for
            spares.
        fudge : float
            Overall fudge factor.

    Notes
    -----
    The blanket and divertor are considered as Operating Costs.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('fusion_island_costing', default=None, types=dict)

    def setup(self):
        self.cc = self.options['fusion_island_costing']
        #  todo: raise a proper error
        self.add_input("P_t", units="MW")
        self.add_input("Cpc", units="GUSD")
        self.add_input("Csg", units="GUSD")
        self.add_input("Cst", units="GUSD")
        self.add_input("C_aux", units="GUSD")

        self.add_output("C_ht", units="GUSD", ref=1)
        self.add_output("C_FI", units="GUSD", ref=1)

    def compute(self, inputs, outputs):
        Pt = inputs['P_t']

        if Pt <= 0:
            raise om.AnalysisError("Pt must be positive.")

        c_Pt = self.cc['c_Pt']
        d_Pt = self.cc['d_Pt']
        e_Pt = self.cc['e_Pt']

        C_ht = c_Pt * (Pt / d_Pt)**e_Pt
        outputs['C_ht'] = C_ht

        Cpc = inputs['Cpc']
        Csg = inputs['Csg']
        Cst = inputs['Cst']
        Caux = inputs['C_aux']
        f = self.cc['fudge']

        C_FI = f * (C_ht + self.cc['m_pc'] * Cpc + self.cc['m_sg'] * Csg +
                    self.cc['m_st'] * Cst + self.cc['m_aux'] * Caux)
        outputs['C_FI'] = C_FI

    def setup_partials(self):
        f = self.cc["fudge"]
        self.declare_partials('C_ht', ['P_t'])
        self.declare_partials('C_FI', ['P_t'])
        self.declare_partials('C_FI', ['Cpc'], val=f * self.cc['m_pc'])
        self.declare_partials('C_FI', ['Csg'], val=f * self.cc['m_sg'])
        self.declare_partials('C_FI', ['Cst'], val=f * self.cc['m_st'])
        self.declare_partials('C_FI', ['C_aux'], val=f * self.cc['m_aux'])

    def compute_partials(self, inputs, J):
        Pt = inputs['P_t']
        c_Pt = self.cc['c_Pt']
        d_Pt = self.cc['d_Pt']
        e_Pt = self.cc['e_Pt']

        f = self.cc["fudge"]

        J["C_ht", "P_t"] = c_Pt * e_Pt * (Pt / d_Pt)**e_Pt / Pt
        J["C_FI", "P_t"] = f * J["C_ht", "P_t"]


class CapitalCost(om.ExplicitComponent):
    r"""Generomak plant capital cost

    From Equation (19) of [1]_,

    .. math::
       C_D = f_{cont}\left(\left(c_{e1} + c_{e2} P_e/c_{e3}\right)
                           \left(P_t / d_{Pt}\right)^{e_{Pt}}
                           + c_V(V_{FI}/d_{V})^{e_{V}}
                           + C_{FI}\right).
    Inputs
    ------
    P_e : float
       MW, Electric power generated by the turbine
    P_t : float
       MW, Thermal power of the system
    V_FI: float
       m**3, volume of the fusion island
    C_FI: float
       GUSD, Cost of the fusion island components

    Outputs
    -------
    C_BOP : float
       GUSD, Capital cost of the balance-of-plant systems,
       including turbine equipment. Does not include contingency.
    C_bld : float
       GUSD, Capital cost of reactor building, hot cells, vacuum systems,
       power supplies and peripherals, and cryogenic systems. Does not
       include contingency.
    C_D : float
       GUSD, Capital cost of the overall plant

    Options
    -------
    capital_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        f_cont : float
            Contigency factor.
        c_e1 : float
            GUSD, Base BOP cost for any plant.
        c_e2 : float
            GUSD, Additional cost for a plant of power c_e3.
        c_e3 : float
            MW, Net Electric power of a typical plant.
        d_Pt : float
            MW, Divisor for the thermal conversion systems cost.
        e_Pt : float
            Exponent for the thermal conversion systems cost.
        c_V : float
            GUSD, Coefficient for the fusion island volume.
        d_V : float
            m^3, Divisor for the fusion island volume.
        e_V : float
            Exponent for the fusion island volume.

    Notes
    -----
    Sheffield treats V_FI as the smallest cylinder that the TF fits inside.
    Something like the cryostat volume might be a suitable replacement, if it's
    not too much larger.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('capital_cost_coeffs', default=None)

    def setup(self):
        self.cc = self.options['capital_cost_coeffs']
        self.add_input("P_e", units="MW")
        self.add_input("P_t", units="MW")
        self.add_input("V_FI", units="m**3")
        self.add_input("C_FI", units="GUSD")

        self.add_output("C_BOP", units="GUSD", ref=1)
        self.add_output("C_bld", units="GUSD", ref=1)
        self.add_output("C_D", units="GUSD", ref=1)

    def compute(self, inputs, outputs):
        Pt = inputs['P_t']
        Pe = inputs['P_e']

        if Pt <= 0:
            raise om.AnalysisError("P_t must be positive.")

        if Pe <= 0:
            raise om.AnalysisError("P_e must be positive.")

        cc = self.cc

        C_BOP = ((cc['c_e1'] + cc['c_e2'] * Pe / cc['c_e3']) *
                 (Pt / cc['d_Pt'])**cc['e_Pt'])
        outputs['C_BOP'] = C_BOP

        V_FI = inputs['V_FI']
        C_bld = cc['c_V'] * (V_FI / cc['d_V'])**cc['e_V']
        outputs["C_bld"] = C_bld

        C_FI = inputs['C_FI']

        C_D = cc['f_cont'] * (C_BOP + C_bld + C_FI)
        outputs['C_D'] = C_D

    def setup_partials(self):
        f_cont = self.cc['f_cont']
        self.declare_partials('C_BOP', ['P_e', 'P_t'])
        self.declare_partials('C_bld', ['V_FI'])
        self.declare_partials('C_D', ['P_e', 'P_t', 'V_FI'])
        self.declare_partials('C_D', ['C_FI'], val=f_cont)

    def compute_partials(self, inputs, J):
        Pt = inputs['P_t']
        Pe = inputs['P_e']

        cc = self.cc
        c_e1 = cc['c_e1']
        c_e2 = cc['c_e2']
        c_e3 = cc['c_e3']

        d_Pt = cc['d_Pt']
        e_Pt = cc['e_Pt']
        f_cont = cc['f_cont']

        J["C_BOP", "P_t"] = ((c_e1 + c_e2 * Pe / c_e3) *
                             (e_Pt * (Pt / d_Pt)**(e_Pt) / Pt))
        J["C_BOP", "P_e"] = (c_e2 / c_e3) * (Pt / d_Pt)**(e_Pt)
        J["C_D", "P_t"] = f_cont * J["C_BOP", "P_t"]
        J["C_D", "P_e"] = f_cont * J["C_BOP", "P_e"]

        c_V = cc['c_V']
        d_V = cc['d_V']
        e_V = cc['e_V']
        V_FI = inputs['V_FI']
        J["C_bld", "V_FI"] = (c_V * e_V * (V_FI / d_V)**e_V) / V_FI
        J["C_D", "V_FI"] = f_cont * J["C_bld", "V_FI"]


class DeuteriumCost(om.ExplicitComponent):
    r"""Similar to the STARFIRE report.

    Assumes a D-T reactor.

    Inputs
    ------
    P_fus: float
       MW, Fusion power
    f_av: float
       Availability factor

    Outputs
    -------
    C_deuterium: float
       MUSD/a, Average annual deuterium cost

    Options
    -------
    deuterium_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        C_deu  : float
            USD/kg, Price of Deuterium in $/kg.

    Notes
    -----
    f_av here is the fraction of time during which the plasma is in flattop.

    For a similar computation see pg 512 of Volume II of the STARFIRE report
    [1]_.

    The price of deuterium is estimated from wikipedia: it lists $13k/kg, but I
    estimated it as $10k/kg.

    References
    ----------
    ..[1] Baker, C. C.; Abdou, M. A. et al.
      STARFIRE - A Commerical Tokamak Fusion Power Study. Volume II.
      ANL/FPP-80-1; Argonne National Laboratory: Argonne, Illinois, 1980.
      https://doi.org/10.2172/6633213
    """
    def initialize(self):
        self.options.declare('deuterium_cost_coeffs', types=dict)

    def setup(self):
        self.cc = self.options['deuterium_cost_coeffs']
        self.add_input("P_fus", units="MW")
        self.add_input("f_av")
        self.add_output("C_deuterium", units="MUSD/a", ref=0.5)
        self.add_output("D usage", units="kg/a", ref=50)

        # this was done by looking up the masses of D, T, He atoms, and
        # neutrons, and doing Δm c²
        self.conversion = 0.0374239  # kg per MW*y

    def compute(self, inputs, outputs):
        cc = self.cc
        P_av = inputs["P_fus"] * inputs["f_av"]
        C_per_kg = cc['C_deu_per_kg']
        outputs["D usage"] = P_av * self.conversion
        outputs["C_deuterium"] = C_per_kg * P_av * self.conversion / mega

    def setup_partials(self):
        self.declare_partials("D usage", ["P_fus", "f_av"])
        self.declare_partials("C_deuterium", ["P_fus", "f_av"])

    def compute_partials(self, inputs, J):
        cc = self.cc
        f_av = inputs["f_av"]
        P_fus = inputs["P_fus"]
        C_per_kg = cc['C_deu_per_kg']
        J["C_deuterium", "P_fus"] = f_av * self.conversion * C_per_kg / mega
        J["C_deuterium", "f_av"] = P_fus * self.conversion * C_per_kg / mega
        J["D usage", "P_fus"] = f_av * self.conversion
        J["D usage", "f_av"] = P_fus * self.conversion


class MiscReplacements(om.ExplicitComponent):
    r"""Miscellaneous replacements

    Inputs
    ------
    C_fuel: float
        MUSD/a, Annual cost of fuel (Deuterium? Lithium?)

    Outputs
    -------
    C_fa : float
        MUSD/a, Cost of miscellaneous scheduled replaceable items

    Options
    -------
    misc_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        C_misc: float
            MUSD, Capital cost of miscellaneous scheduled-replaceable items.

    Notes
    -----
    The original paper uses C_fa = 0.4 + 24 f_CR0. The new paper [1]_ uses an
    inflation factor of 2.19 from 1983 to 2010, but also says C_fa = 7.5 M
    with f_CR0 = 0.078. I'm not sure how this is consistent.

    Here the default C_misc is from multiplying 24 * 2.2.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('misc_cost_coeffs', types=dict)

    def setup(self):
        self.cc = self.options['misc_cost_coeffs']
        self.add_input("C_fuel", units="MUSD/a", val=0.8)
        self.add_output("C_fa", units="MUSD/a", ref=10)

    def compute(self, inputs, outputs):
        cc = self.cc
        f_CR0 = cc['f_CR0']
        C_misc = cc['C_misc']
        C_fuel = inputs['C_fuel']
        C_fa = C_fuel + f_CR0 * C_misc
        outputs['C_fa'] = C_fa

    def setup_partials(self):
        self.declare_partials('C_fa', 'C_fuel', val=1)


class FuelCycleCost(om.ExplicitComponent):
    r"""Fuel cycle cost sum

    Inputs
    ------
    C_ba : float
        MUSD/a, Averaged blanket costs
    C_ta : float
        MUSD/a, Averaged divertor costs
    C_aa : float
        MUSD/a, Averaged aux heating costs
    C_fa : float
        MUSD/a, Fuel costs

    Outputs
    -------
    C_F : float
        GUSD/a, Fuel cycle cost. Includes all blanket and divertor replacement
        and C_OM.

    From Equation (22) of [1]_.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def setup(self):
        self.add_input("C_ba", units="MUSD/a", val=0.0)
        self.add_input("C_ta", units="MUSD/a", val=0.0)
        self.add_input("C_aa", units="MUSD/a", val=0.0)
        self.add_input("C_fa", units="MUSD/a", val=0.0)
        self.add_output("C_F", units="GUSD/a", ref=0.100)

    def compute(self, inputs, outputs):
        outputs["C_F"] = (inputs["C_ba"] + inputs["C_ta"] + inputs["C_aa"] +
                          inputs["C_fa"]) / 1000

    def setup_partials(self):
        self.declare_partials("C_F", ["C_ba", "C_ta", "C_aa", "C_fa"],
                              val=0.001)


class AveragedAnnualBlanketCost(om.ExplicitComponent):
    r"""
    From Equation (23) of [1]_.

    .. math::
        C_{ba} = f_{fail} \left(f_{spare} C_{bl} F_{CR0}
                        + (f_{av} N p_{wn} / F_{wn} - 1)C_bl/N)

    where :math:`N` is the number of years of plant operation.

    Inputs
    ------
    C_bl : float
        MUSD, cost of the initial blanket
    f_av : float
        Plant availability
    p_wn : float
        MW/m**2, Neutron wall loading
    F_wn : float
        First wall and blanket lifetime
    N_years : float
        Number of years of plant operation

    Outputs
    -------
    C_ba : float
        MUSD, Averaged annual blanket costs
    Avg blanket repl : float
        MUSD, Averaged annual blanket replacement costs

    blanket_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        f_failures : float
            Contigency factor for failures.
        f_spares : float
            Factor for spares.
        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('blanket_cost_coeffs', types=dict)

    def setup(self):
        self.cc = self.options['blanket_cost_coeffs']
        self.add_input("C_bl", units="MUSD", val=0.0)
        self.add_input("f_av", val=1.0)
        self.add_input("F_wn", units="MW*a/m**2")
        self.add_input("p_wn", units="MW/m**2")
        self.add_input("N_years")
        self.add_output("C_ba", units="MUSD/a", lower=0, ref=100)
        self.add_output("Initial blanket", units="MUSD", lower=0, ref=100)
        self.add_output("Avg blanket repl", units="MUSD/a", lower=0, ref=100)

    def compute(self, inputs, outputs):
        cc = self.cc
        C_bl = inputs["C_bl"]
        f_av = inputs["f_av"]
        F_wn = inputs["F_wn"]
        p_wn = inputs["p_wn"]
        n_years = inputs["N_years"]

        initial_bl = cc['f_spares'] * C_bl * cc['F_CR0']
        outputs["Initial blanket"] = initial_bl
        # averaged cost of scheduled blanket replacement
        avg_sched_repl = (f_av * n_years * p_wn / F_wn - 1) * C_bl / n_years
        outputs["Avg blanket repl"] = avg_sched_repl
        outputs["C_ba"] = cc['f_failures'] * (initial_bl +
                                              avg_sched_repl) * cc['fudge']

    def setup_partials(self):
        cc = self.cc
        self.declare_partials("Initial blanket", ["C_bl"],
                              val=cc['F_CR0'] * cc['f_spares'])
        self.declare_partials("Avg blanket repl",
                              ["C_bl", "f_av", "p_wn", "F_wn", "N_years"])
        self.declare_partials("C_ba",
                              ["C_bl", "f_av", "p_wn", "F_wn", "N_years"])

    def compute_partials(self, inputs, J):
        cc = self.cc
        C_bl = inputs["C_bl"]
        f_av = inputs["f_av"]
        F_wn = inputs["F_wn"]
        p_wn = inputs["p_wn"]
        n_years = inputs["N_years"]

        J["Avg blanket repl", "C_bl"] = -1 / n_years + f_av * p_wn / F_wn
        J["Avg blanket repl", "f_av"] = C_bl * p_wn / F_wn
        J["Avg blanket repl", "N_years"] = C_bl / n_years**2
        J["Avg blanket repl", "p_wn"] = C_bl * f_av / F_wn
        J["Avg blanket repl", "F_wn"] = -C_bl * f_av * p_wn / F_wn**2

        ffails = cc['f_failures'] * cc['fudge']

        J["C_ba", "C_bl"] = ffails * (J["Initial blanket", "C_bl"] +
                                      J["Avg blanket repl", "C_bl"])
        J["C_ba", "f_av"] = ffails * J["Avg blanket repl", "f_av"]
        J["C_ba", "N_years"] = ffails * J["Avg blanket repl", "N_years"]
        J["C_ba", "p_wn"] = ffails * J["Avg blanket repl", "p_wn"]
        J["C_ba", "F_wn"] = ffails * J["Avg blanket repl", "F_wn"]


class AveragedAnnualDivertorCost(om.ExplicitComponent):
    r"""
    From Equation (24) of [1]_.

    .. math::
        C_{ta} = f_{fail} \left(f_{spare} C_{tt} F_{CR0}
                        + (f_{av} N p_{tt} / F_{tt} - 1)C_tt/N)

    where :math:`N` is the number of years of plant operation.

    Inputs
    ------
    C_tt : float
        MUSD, cost of the initial divertor
    f_av : float
        Plant availability
    p_tt : float
        MW/m**2, Averaged thermal load on the divertor targets
    F_tt : float
        MW*a/m**2, Divertor lifetime
    N_years : float
        Number of years of plant operation

    Outputs
    -------
    C_ba : float
        MUSD/a, Averaged divertor costs
    Avg divertor repl : float
        MUSD/a, Averaged divertor replacement costs

    divertor_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        f_failures : float
            Contigency factor for failures.
        f_spares : float
            Factor for spares.
        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('divertor_cost_coeffs', default=None)

    def setup(self):
        self.cc = self.options['divertor_cost_coeffs']
        self.add_input("C_tt", units="MUSD", val=0.0)
        self.add_input("f_av", val=1.0)
        self.add_input("F_tt", units="MW*a/m**2")
        self.add_input("p_tt", units="MW/m**2")
        self.add_input("N_years")
        self.add_output("C_ta", units="MUSD/a", lower=0, ref=100)
        self.add_output("Initial divertor", units="MUSD", lower=0, ref=100)
        self.add_output("Avg divertor repl", units="MUSD/a", lower=0, ref=100)

    def compute(self, inputs, outputs):
        cc = self.cc
        C_tt = inputs["C_tt"]
        f_av = inputs["f_av"]
        F_tt = inputs["F_tt"]
        p_tt = inputs["p_tt"]
        n_years = inputs["N_years"]

        initial_tt = cc['f_spares'] * C_tt * cc['F_CR0']
        outputs["Initial divertor"] = initial_tt
        # averaged cost of scheduled divertor replacement
        avg_sched_repl = (f_av * n_years * p_tt / F_tt - 1) * C_tt / n_years
        outputs["Avg divertor repl"] = avg_sched_repl
        outputs["C_ta"] = cc['f_failures'] * (initial_tt +
                                              avg_sched_repl) * cc['fudge']

    def setup_partials(self):
        cc = self.cc
        self.declare_partials("Initial divertor", ["C_tt"],
                              val=cc['F_CR0'] * cc['f_spares'])
        self.declare_partials("Avg divertor repl",
                              ["C_tt", "f_av", "p_tt", "F_tt", "N_years"])
        self.declare_partials("C_ta",
                              ["C_tt", "f_av", "p_tt", "F_tt", "N_years"])

    def compute_partials(self, inputs, J):
        cc = self.cc
        C_tt = inputs["C_tt"]
        f_av = inputs["f_av"]
        F_tt = inputs["F_tt"]
        p_tt = inputs["p_tt"]
        n_years = inputs["N_years"]

        J["Avg divertor repl", "C_tt"] = -1 / n_years + f_av * p_tt / F_tt
        J["Avg divertor repl", "f_av"] = C_tt * p_tt / F_tt
        J["Avg divertor repl", "N_years"] = C_tt / n_years**2
        J["Avg divertor repl", "p_tt"] = C_tt * f_av / F_tt
        J["Avg divertor repl", "F_tt"] = -C_tt * f_av * p_tt / F_tt**2

        ffails = cc['f_failures'] * cc['fudge']

        J["C_ta", "C_tt"] = ffails * (J["Initial divertor", "C_tt"] +
                                      J["Avg divertor repl", "C_tt"])
        J["C_ta", "f_av"] = ffails * J["Avg divertor repl", "f_av"]
        J["C_ta", "N_years"] = ffails * J["Avg divertor repl", "N_years"]
        J["C_ta", "p_tt"] = ffails * J["Avg divertor repl", "p_tt"]
        J["C_ta", "F_tt"] = ffails * J["Avg divertor repl", "F_tt"]


class FixedOMCost(om.ExplicitComponent):
    r"""From Sheffield; scaled based off STARFIRE.

    Inputs
    ------
    P_e : float
        MW, net electric power

    Outputs
    -------
    C_OM : float
        MUSD/a

    Options
    -------
    fixed_om_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        base_Pe  : float
            MW, Base electric power for the scaling formula
        base_OM  : float
            MUSD/a, Fixed operations and maintenance cost for a plant that
            generates base_Pe of net electric power.
        fudge  : float
            Overall fudge factor.

    Notes
    -----
    It appears that this value assumes a steam plant.
    See Equation (F.10) of [2]_.

    It is based on numbers from STARFIRE, a 1200MWe plant.
    This includes staff costs, annual misc. consumables and equipment,
    annual outside support services, annual general and admin costs,
    annual coolant makeup (=0), annual process material (for water and T)
    processing, annual fuel handling costs (=0), and annual miscellaneous costs
    (training, requalification of operators, rent of equipment, travel).

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.

    .. [2] Sheffield, J. et al.
       Cost Assessment of a Generic Magnetic Fusion Reactor.
       Fusion Technology 1986, 9 (2), 199–249.
       https://doi.org/10.13182/FST9-2-199.
    """
    def initialize(self):
        self.options.declare("fixed_om_cost_coeffs", types=dict)

    def setup(self):
        self.cc = self.options["fixed_om_cost_coeffs"]
        self.add_input("P_e", units="MW")
        self.add_output("C_OM", units="MUSD/a", ref=100)

    def compute(self, inputs, outputs):
        cc = self.cc
        Pe = inputs["P_e"]

        if Pe <= 0:
            raise om.AnalysisError("Net electric power must be positive")

        base_OM = cc["base_OM"]
        base_Pe = cc["base_Pe"]
        fudge = cc["fudge"]
        cost = base_OM * (Pe / base_Pe)**(1 / 2)
        outputs["C_OM"] = fudge * cost

    def setup_partials(self):
        self.declare_partials("C_OM", "P_e")

    def compute_partials(self, inputs, J):
        cc = self.cc
        Pe = inputs["P_e"]

        if Pe <= 0:
            raise om.AnalysisError("Net electric power must be positive")

        base_OM = cc["base_OM"]
        base_Pe = cc["base_Pe"]
        fudge = cc["fudge"]
        J["C_OM", "P_e"] = fudge * base_OM / (2 * (base_Pe * Pe)**(1 / 2))


class TotalCapitalCost(om.ExplicitComponent):
    r"""
    Equation (27) of [1]_.

    Inputs
    ------
    C_D : float
        GUSD, overnight capital cost of the plant
    f_CAP0 : float
        Constant-dollar capitalization factor
    f_IND : float
        Factor for indirect charges

    Outputs
    -------
    C_C0 : float
        GUSD, Total capital cost of the plant

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def setup(self):
        self.add_input("C_D", units="GUSD")
        self.add_input("f_CAP0", val=1.0)
        self.add_input("f_IND", val=1.0)
        self.add_output("C_C0", units="GUSD", ref=3)

    def compute(self, inputs, outputs):
        C_D = inputs["C_D"]
        f_CAP0 = inputs["f_CAP0"]
        f_IND = inputs["f_IND"]
        outputs["C_C0"] = C_D * f_CAP0 * f_IND

    def setup_partials(self):
        self.declare_partials("C_C0", ["C_D", "f_CAP0", "f_IND"])

    def compute_partials(self, inputs, J):
        C_D = inputs["C_D"]
        f_CAP0 = inputs["f_CAP0"]
        f_IND = inputs["f_IND"]
        J["C_C0", "C_D"] = f_CAP0 * f_IND
        J["C_C0", "f_CAP0"] = C_D * f_IND
        J["C_C0", "f_IND"] = C_D * f_CAP0


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

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def setup(self):
        self.add_input("T_constr", units="a")
        self.add_output("f_IND")

    def compute(self, inputs, outputs):
        T_constr = inputs["T_constr"]
        f_ind = 1 + 0.5 * T_constr / 8
        outputs["f_IND"] = f_ind

    def setup_partials(self):
        self.declare_partials("f_IND", "T_constr", val=0.5 / 8)


class ConstantDollarCapitalizationFactor(om.ExplicitComponent):
    r"""

    Table IV of [1]_.

    Inputs
    ------
    T_constr : float
        a, Construction time

    Outputs
    -------
    f_CAP0 : float
        Constant-dollar capitalization factor

    Notes
    -----
    The table data is linear up to 10 years, then decreases slope between 10
    and 12 years. I ignore that dogleg.

    These values apparently correspond to rates of 0.06
    for inflation and escalation.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def setup(self):
        self.add_input("T_constr", units='a')
        self.add_output("f_CAP0")

    def compute(self, inputs, outputs):
        T = inputs["T_constr"]
        f_cap0 = 1 + 0.012 * T + 0.003
        outputs["f_CAP0"] = f_cap0

    def setup_partials(self):
        self.declare_partials("f_CAP0", "T_constr", val=0.012)


class CostOfElectricity(om.ExplicitComponent):
    r"""
    Equation (26) of [1]_.

    Inputs
    ------
    C_C0 : float
        MUSD, total capital cost
    C_F : float
        MUSD, Annual fuel cycle costs
    C_OM : float
        mUSD/kW/h, Operations and maintenance cost
    f_av : float
        Availability factor
    P_e : float
        MW, Electric power output

    Outputs
    -------
    COE : float
        mUSD/kW/h, Cost of electricity

    Options
    -------
    coe_cost_coeffs : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor.
        waste_charge : float
            mUSD/kW/h, Cost of waste disposal

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def initialize(self):
        self.options.declare('coe_cost_coeffs', default=None)

    def setup(self):
        self.cc = self.options['coe_cost_coeffs']
        self.add_input("C_C0", units="MUSD", val=0.0)
        self.add_input("C_F", units="MUSD/a", val=0.0)
        self.add_input("C_OM", units="MUSD/a", val=0.0)
        self.add_input("f_av", val=1.0)
        self.add_input("P_e", units="MW")

        self.add_output("COE", units="mUSD/kW/h", lower=0, ref=100)

    def compute(self, inputs, outputs):
        cc = self.cc
        C_C0 = inputs["C_C0"]
        C_F = inputs["C_F"]
        C_OM = inputs["C_OM"]
        Pe = inputs["P_e"]
        f_av = inputs["f_av"]

        numerator = mega * (C_C0 * cc['F_CR0'] + C_F + C_OM)
        h_per_y = 8760
        electricity_produced = Pe * h_per_y * f_av
        coe = numerator / electricity_produced + cc['waste_charge']
        outputs["COE"] = cc['fudge'] * coe

    def setup_partials(self):
        self.declare_partials("COE", ["C_C0", "C_F", "C_OM", "P_e", "f_av"])

    def compute_partials(self, inputs, J):
        cc = self.cc
        C_C0 = inputs["C_C0"]
        C_F = inputs["C_F"]
        C_OM = inputs["C_OM"]
        Pe = inputs["P_e"]
        f_av = inputs["f_av"]
        F_CR0 = cc['F_CR0']
        h_per_y = 8760
        fudge = cc['fudge']

        J["COE", "C_C0"] = fudge * F_CR0 * mega / (h_per_y * f_av * Pe)
        J["COE", "C_F"] = fudge * mega / (h_per_y * f_av * Pe)
        J["COE", "C_OM"] = fudge * mega / (h_per_y * f_av * Pe)
        J["COE", "P_e"] = -fudge * mega * ((C_F + C_OM + C_C0 * F_CR0) /
                                           (h_per_y * f_av * Pe**2))
        J["COE", "f_av"] = -fudge * mega * ((C_F + C_OM + C_C0 * F_CR0) /
                                            (h_per_y * f_av**2 * Pe))


class GeneromakStructureVolume(om.ExplicitComponent):
    r"""Magnet structure volume as a constant * magnet volume

    .. math::

       V_{st} = 0.75 V_{pc}

    Inputs
    ------
    V_pc: float
       m**3, primary coil volume

    Outputs
    -------
    V_st: float
       m**3, Magnet structure volume

    Notes
    -----
    From Equation (17) of [1]_.

    References
    ----------
    .. [1] Sheffield, J.; Milora, S. L.
       Generic Magnetic Fusion Reactor Revisited.
       Fusion Science and Technology 2016, 70 (1), 14–35.
       https://doi.org/10.13182/FST15-157.
    """
    def setup(self):
        self.add_input("V_pc", units='m**3', val=0.0)
        self.add_output("V_st", units='m**3', ref=100)
        self.c = 0.75

    def compute(self, inputs, outputs):
        outputs["V_st"] = self.c * inputs['V_pc']

    def setup_partials(self):
        self.declare_partials('V_st', 'V_pc', val=self.c)


class GeneromakCosting(om.Group):
    r"""
    Inputs
    ------
    V_FI: float
       m**3, volume of the fusion island
    V_pc: float
       m**3, Material volume of primary (TF) coils
    V_sg: float
       m**3, Material volume of the shield (with gaps)
    V_st: float
       m**3, Material volume of the inter-coil structures and gravity supports.
    V_bl: float
       m**3, Material volume of the blanket
    A_tt: float
       m**2, Wall area taken up by divertor

    P_aux: float
       MW, power of aux heating systems
    P_e : float
       MW, Electric power generated by the turbine
    P_t : float
       MW, Thermal power of the system
    P_fus: float
       MW, Fusion power

    p_wn : float
        MW/m**2, Neutron wall loading
    F_wn : float
        First wall and blanket lifetime
    p_tt : float
        MW/m**2, Averaged thermal load on the divertor targets
    F_tt : float
        MW*a/m**2, Divertor lifetime

    f_av : float
        Plant availability
    T_constr : float
        a, Construction time
    N_years : float
        Number of years of plant operation

    Outputs
    -------
    COE: float
        mUSD/kW/h, cost of electricity

    Options
    -------
    exact_generomak: bool
        If true, then
        1. The plant capital and O&M costs use the net electric power.
        2. The coil structure volume is 0.75 of the primary
           coil structure volume.
        3. The fuel cost is $7.5M
        Else,
        1. The plant capital and O&M costs use the gross electric power.
        2. The coil structure volume is an input.
        3. The calculated fuel cost includes a calculation for the price of
           deuterium.

        For either option, the input interface is the same.
    """
    def initialize(self):
        self.options.declare('exact_generomak', default=True)
        self.options.declare('costing_parameters', default=None)

    def setup(self):
        exact_generomak = self.options['exact_generomak']

        default_cost_pars = {
            "Finance.F_CR0": 0.078,
            "Deuterium.cost_per_mass": 10000,  # USD/kg
            "PrimaryCoils.cost_per_vol": 1.66,  # MUSD/m^3
            "Blanket.cost_per_vol": 0.75,  # MUSD/m^3
            "Blanket.f_failures": 1.1,
            "Blanket.f_spares": 1.1,
            "Blanket.fudge": 1.0,
            "Divertor.cost_per_area": 0.114,  # MUSD/m^2
            "Divertor.f_failures": 1.2,
            "Divertor.f_spares": 1.1,
            "Divertor.fudge": 1.0,
            "Structure.cost_per_vol": 0.36,  # MUSD/m^3
            "ShieldWithGaps.cost_per_vol": 0.29,  # MUSD/m^3
            "AuxHeating.cost_per_watt": 5.3,  # MUSD/W
            "AuxHeating.f_spares": 1.1,
            "AuxHeating.ann_maint_fact": 0.1,
            "ReferencePlant.d_Pt": 4150,  # MW
            "ReferencePlant.e_Pt": 0.6,
            "ReferencePlant.d_Pe": 1200,  # MW
            "FusionIsland.c_Pt": 0.221,  # GUSD
            "FusionIsland.m_pc": 1.5,
            "FusionIsland.m_sg": 1.25,
            "FusionIsland.m_st": 1.0,
            "FusionIsland.fudge": 1.0,
            "CapitalCost.contingency": 1.15,
            "CapitalCost.c_e1": 0.900,  # GUSD
            "CapitalCost.c_e2": 0.900,  # GUSD
            "CapitalCost.c_V": 0.839,  # GUST
            "CapitalCost.d_V": 5100,  # m^3
            "CapitalCost.e_V": 0.67,
            "CapitalCost.fudge": 1.0,
            "MiscReplacements.c_misc": 52.8,  # MUSD/a
            "FixedOM.base_cost": 108,  # MUSD/a
            "FixedOM.fudge": 1.0,
            "COE.waste_charge": 0.5,  # mUSD/kW/h
            "COE.fudge": 1.0
        }

        cost_pars = self.options['costing_parameters']
        # replace any updates
        cost_pars = default_cost_pars

        pc_cost_per_vol = cost_pars["PrimaryCoils.cost_per_vol"]
        self.add_subsystem("primary_coilset",
                           PrimaryCoilSetCost(cost_per_volume=pc_cost_per_vol),
                           promotes_inputs=["V_pc"],
                           promotes_outputs=["Cpc"])

        if exact_generomak:
            self.add_subsystem("structure_vol",
                               GeneromakStructureVolume(),
                               promotes_inputs=["V_pc"])
            self.connect("structure_vol.V_st", "coil_structure.V_st")

            # ignore promotes V_st
            self.add_subsystem("ignore",
                               om.ExecComp(["ignore=V_st"],
                                           V_st={'units': 'm**3'}),
                               promotes_inputs=["V_st"])

        sg_cost_per_vol = cost_pars["ShieldWithGaps.cost_per_vol"]
        self.add_subsystem("shielding",
                           ShieldWithGapsCost(cost_per_volume=sg_cost_per_vol),
                           promotes_inputs=["V_sg"],
                           promotes_outputs=["Csg"])
        bl_cost_per_vol = cost_pars["Blanket.cost_per_vol"]
        self.add_subsystem("blanket",
                           BlanketCost(cost_per_volume=bl_cost_per_vol),
                           promotes_inputs=["V_bl"],
                           promotes_outputs=["C_bl"])
        dv_cost_per_area = cost_pars["Divertor.cost_per_area"]
        self.add_subsystem("divertor",
                           DivertorCost(cost_per_area=dv_cost_per_area),
                           promotes_inputs=["A_tt"],
                           promotes_outputs=["C_tt"])

        st_cost_per_vol = cost_pars["Structure.cost_per_vol"]
        self.add_subsystem("coil_structure",
                           StructureCost(cost_per_volume=st_cost_per_vol),
                           promotes_outputs=["Cst"])
        # In the exact generomak, the structure volume is calculated.
        if not exact_generomak:
            self.promotes("coil_structure", inputs=["V_st"])

        aux_cost_per_w = cost_pars["AuxHeating.cost_per_watt"]
        self.add_subsystem("aux_h",
                           AuxHeatingCost(cost_per_watt=aux_cost_per_w),
                           promotes_inputs=["P_aux"],
                           promotes_outputs=["C_aux"])

        # Assume the only 'fuel' is deuterium. Note: this leaves out Li, but
        # that's considered to be part of the blankets(?).
        deu_cc = {"C_deu_per_kg": cost_pars["Deuterium.cost_per_mass"]}
        self.add_subsystem("ann_d_cost",
                           DeuteriumCost(deuterium_cost_coeffs=deu_cc),
                           promotes_inputs=["P_fus", "f_av"])

        if exact_generomak:
            # The 2016 Generomak paper uses a constant 7.5M for miscellaneous
            # costs.
            ivc = om.IndepVarComp()
            ivc.add_output("C_fa", units="MUSD/a", val=7.5)
            self.add_subsystem("misc", ivc, promotes_outputs=["C_fa"])
        else:
            # This allows a variation due to the price of buying deuterium.
            misc_cc = {
                'f_CR0': cost_pars["Finance.F_CR0"],
                'C_misc': cost_pars["MiscReplacements.c_misc"]
            }
            self.add_subsystem("misc",
                               MiscReplacements(misc_cost_coeffs=misc_cc),
                               promotes_outputs=["C_fa"])
            self.connect("ann_d_cost.C_deuterium", "misc.C_fuel")

        ann_aux_cc = {
            'f_spares': cost_pars["AuxHeating.f_spares"],
            'annual_aux_heating_factor': cost_pars["AuxHeating.ann_maint_fact"]
        }
        self.add_subsystem(
            "annual_aux",
            AnnualAuxHeatingCost(annual_aux_heating_costing=ann_aux_cc),
            promotes_inputs=["C_aux"],
            promotes_outputs=["C_aa"])

        fi_cc = {
            'c_Pt': cost_pars["FusionIsland.c_Pt"],
            'd_Pt': cost_pars["ReferencePlant.d_Pt"],
            'e_Pt': cost_pars["ReferencePlant.e_Pt"],
            'm_pc': cost_pars["FusionIsland.m_pc"],
            'm_sg': cost_pars["FusionIsland.m_sg"],
            'm_st': cost_pars["FusionIsland.m_st"],
            'm_aux': cost_pars["AuxHeating.f_spares"],
            'fudge': cost_pars["FusionIsland.fudge"],
        }
        self.add_subsystem(
            "fusion_island",
            FusionIslandCost(fusion_island_costing=fi_cc),
            promotes_inputs=["P_t", "Cpc", "Csg", "Cst", "C_aux"],
            promotes_outputs=["C_FI"])

        cap_cc = {
            'f_cont': cost_pars["CapitalCost.contingency"],
            'c_e1': cost_pars["CapitalCost.c_e1"],
            'c_e2': cost_pars["CapitalCost.c_e2"],
            'c_e3': cost_pars["ReferencePlant.d_Pe"],
            'd_Pt': cost_pars["ReferencePlant.d_Pt"],
            'e_Pt': cost_pars["ReferencePlant.e_Pt"],
            'c_V': cost_pars["CapitalCost.c_V"],
            'd_V': cost_pars["CapitalCost.d_V"],
            'e_V': cost_pars["CapitalCost.e_V"],
            'fudge': cost_pars["CapitalCost.fudge"],
        }
        self.add_subsystem("plant_capital",
                           CapitalCost(capital_cost_coeffs=cap_cc),
                           promotes_inputs=["C_FI", "V_FI", "P_t"],
                           promotes_outputs=["C_D"])
        self.add_subsystem("capitalization_factor",
                           ConstantDollarCapitalizationFactor(),
                           promotes_inputs=["T_constr"],
                           promotes_outputs=["f_CAP0"])
        self.add_subsystem("indirect_charges",
                           IndirectChargesFactor(),
                           promotes_inputs=["T_constr"],
                           promotes_outputs=["f_IND"])
        self.add_subsystem("total_capital",
                           TotalCapitalCost(),
                           promotes_inputs=["C_D", "f_CAP0", "f_IND"],
                           promotes_outputs=["C_C0"])

        ann_bl_cc = {
            'f_failures': cost_pars["Blanket.f_failures"],
            'f_spares': cost_pars["Blanket.f_spares"],
            'F_CR0': cost_pars["Finance.F_CR0"],
            'fudge': cost_pars["Blanket.fudge"],
        }
        self.add_subsystem(
            "ann_blanket_cost",
            AveragedAnnualBlanketCost(blanket_cost_coeffs=ann_bl_cc),
            promotes_inputs=["f_av", "N_years", "p_wn", "C_bl", "F_wn"],
            promotes_outputs=["C_ba"])

        ann_dv_cc = {
            'f_failures': cost_pars["Divertor.f_failures"],
            'f_spares': cost_pars["Divertor.f_spares"],
            'F_CR0': cost_pars["Finance.F_CR0"],
            'fudge': cost_pars["Divertor.fudge"],
        }
        self.add_subsystem(
            "ann_divertor_cost",
            AveragedAnnualDivertorCost(divertor_cost_coeffs=ann_dv_cc),
            promotes_inputs=["f_av", "N_years", "p_tt", "C_tt", "F_tt"],
            promotes_outputs=["C_ta"])

        self.add_subsystem("fuel_cycle_cost",
                           FuelCycleCost(),
                           promotes_inputs=["C_ba", "C_ta", "C_aa", "C_fa"],
                           promotes_outputs=["C_F"])
        fom_cc = {
            "base_Pe": cost_pars["ReferencePlant.d_Pe"],
            "base_OM": cost_pars["FixedOM.base_cost"],
            "fudge": cost_pars["FixedOM.fudge"],
        }
        self.add_subsystem("omcost",
                           FixedOMCost(fixed_om_cost_coeffs=fom_cc),
                           promotes_outputs=["C_OM"])

        if exact_generomak:
            # use the _net_ electric generation only. This ignores the cost of
            # capital to generate the recirculating power.
            self.promotes("plant_capital", inputs=[("P_e", "P_net")])
            self.promotes("omcost", inputs=[("P_e", "P_net")])
            # create a stub so that P_e has something to connect to.
            self.add_subsystem("ignore2",
                               om.ExecComp(["ignore=P_e"], P_e={'units':
                                                                'MW'}),
                               promotes_inputs=["P_e"])
        else:
            self.promotes("plant_capital", inputs=["P_e"])
            self.promotes("omcost", inputs=["P_e"])

        coe_cc = {
            'F_CR0': cost_pars["Finance.F_CR0"],
            'waste_charge': cost_pars["COE.waste_charge"],
            'fudge': cost_pars["COE.fudge"],
        }
        self.add_subsystem(
            "coe",
            CostOfElectricity(coe_cost_coeffs=coe_cc),
            promotes_inputs=["C_C0", "C_F", ("P_e", "P_net"), "f_av", "C_OM"],
            promotes_outputs=["COE"])


if __name__ == '__main__':
    prob = om.Problem()

    prob.model = GeneromakCosting(exact_generomak=True)

    prob.setup(force_alloc_complex=True)

    # geometry
    prob.set_val("V_FI", 5152, units='m**3')
    prob.set_val("V_pc", 607, units='m**3')
    prob.set_val("V_sg", 914, units='m**3')
    prob.set_val("V_st", 607 * 0.75, units='m**3')
    prob.set_val("V_bl", 321, units='m**3')
    prob.set_val("A_tt", 60, units='m**2')

    # # power levels
    prob.set_val("P_aux", 50, units='MW')
    prob.set_val("P_fus", 2250, units="MW")
    prob.set_val("P_t", 2570, units="MW")
    prob.set_val("P_e", 1157, units="MW")
    prob.set_val("P_net", 1000, units="MW")

    # # blanket and divertor power fluxes
    prob.set_val("p_wn", 3, units='MW/m**2')
    prob.set_val("p_tt", 10, units="MW/m**2")
    prob.set_val("F_wn", 15, units="MW*a/m**2")
    prob.set_val("F_tt", 10, units="MW*a/m**2")

    # # fraction of year with fusion
    prob.set_val("f_av", 0.8)

    # # plant lifetime
    prob.set_val("N_years", 30)

    # # construction time
    prob.set_val("T_constr", 6, units='a')

    prob.run_driver()
    prob.model.list_inputs(units=True)
    prob.model.list_outputs(units=True)
