from faroes.configurator import UserConfigurator

import openmdao.api as om
from scipy.constants import mega, hour, year, kilo


class PrimaryCoilSetCost(om.ExplicitComponent):
    r"""Generomak primary TF coil set cost

    .. math::
       C_{pc} = \mathrm{cost}\,V_{pc}

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
    From the last column of Table III of :footcite:t:`sheffield_generic_2016`
    This uses the Adjusted for ITER Unit Cost values.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_pc",
                       units="m**3",
                       val=0.0,
                       desc="Material volume of primary (TF) coils")
        self.add_output("Cpc",
                        units="MUSD",
                        ref=100,
                        desc="Cost of primary coilset")

    def compute(self, inputs, outputs):
        outputs['Cpc'] = self.cpv * inputs["V_pc"]

    def setup_partials(self):
        self.declare_partials('Cpc', ['V_pc'], val=self.cpv)


class BlanketCost(om.ExplicitComponent):
    r"""Generomak blanket set cost

    .. math::
       C_{bl} = \mathrm{cost}\,V_{bl}

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
    From the last column of Table III of :footcite:t:`sheffield_generic_2016`.
    This uses the Adjusted for ITER Unit Cost values.

    In this model, the blanket is replaced, so it is considered as an operating
    cost rather than as a capital cost.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_bl", units="m**3", desc="Blanket volume")
        self.add_output("C_bl",
                        units="MUSD",
                        ref=100,
                        desc="Cost of one blanket set")

    def compute(self, inputs, outputs):
        outputs['C_bl'] = self.cpv * inputs["V_bl"]

    def setup_partials(self):
        self.declare_partials('C_bl', ['V_bl'], val=self.cpv)


class StructureCost(om.ExplicitComponent):
    r"""Generomak structure cost

    .. math:: C_{st} = \mathrm{cost} \, V_{st}

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
    From the last column of Table III of :footcite:t:`sheffield_generic_2016`.
    This uses the Adjusted for ITER Unit Cost values.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_st",
                       units="m**3",
                       desc="Material volume of coil structures & supports")
        self.add_output("Cst",
                        units="MUSD",
                        ref=100,
                        desc="Cost of coil structures & supports")

    def compute(self, inputs, outputs):
        outputs['Cst'] = self.cpv * inputs["V_st"]

    def setup_partials(self):
        self.declare_partials('Cst', ['V_st'], val=self.cpv)


class ShieldWithGapsCost(om.ExplicitComponent):
    r"""Generomak shield with gaps cost

    .. math:: C_{sg} = \mathrm{cost} \, V_{sg}

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
    From the last column of Table III of :footcite:t:`sheffield_generic_2016`.
    This uses the Adjusted for ITER Unit Cost values.
    """
    def initialize(self):
        self.options.declare('cost_per_volume')

    def setup(self):
        self.cpv = self.options['cost_per_volume']
        self.add_input("V_sg", units="m**3", desc="Volume of shield with gaps")
        self.add_output("Csg",
                        units="MUSD",
                        ref=100,
                        desc="Cost of shielding with gaps")

    def compute(self, inputs, outputs):
        outputs['Csg'] = self.cpv * inputs["V_sg"]

    def setup_partials(self):
        self.declare_partials('Csg', ['V_sg'], val=self.cpv)


class DivertorCost(om.ExplicitComponent):
    r"""Generomak replaceable divertor cost

    .. math:: C_{tt} = \mathrm{cost} \, V_{tt}

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
    From the paragraph below Equation (24) of
    :footcite:t:`sheffield_generic_2016`.
    The Generomak estimates the target area as 10% of the wall area,
    and estimates a thermal load on the divertor targets of 10 MW/m^2
    when the neutron wall load p_wn is 3.0 MW/m^2. This doesn't seem compatible
    with modern ideas to have detached or very radiative divertors.
    """
    def initialize(self):
        self.options.declare('cost_per_area')

    def setup(self):
        self.cpa = self.options['cost_per_area']
        self.add_input("A_tt", units="m**2", desc="Divertor wall area")
        self.add_output("C_tt", units="MUSD", ref=3, desc="Divertor set cost")

    def compute(self, inputs, outputs):
        outputs['C_tt'] = self.cpa * inputs["A_tt"]

    def setup_partials(self):
        self.declare_partials('C_tt', ['A_tt'], val=self.cpa)


class AuxHeatingCost(om.ExplicitComponent):
    r"""Generomak auxilliary heating cost

    .. math:: C_{aux} = \mathrm{cost} P_{aux}

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
    III of :footcite:t:`sheffield_generic_2016`.
    """
    def initialize(self):
        self.options.declare('cost_per_watt')

    def setup(self):
        self.cpw = self.options['cost_per_watt']
        self.add_input("P_aux",
                       units="MW",
                       desc="Power of aux heating systems")
        self.add_output("C_aux",
                        units="MUSD",
                        ref=100,
                        desc="Cost of aux heating systems")

    def compute(self, inputs, outputs):
        outputs['C_aux'] = self.cpw * inputs["P_aux"]

    def setup_partials(self):
        self.declare_partials('C_aux', ['P_aux'], val=self.cpw)


class AnnualAuxHeatingCost(om.ExplicitComponent):
    r"""Part of the 'fuel cycle costs'

    .. math:: C_{aa} = f_{spares} f_{annual} C_{aux}

    Inputs
    ------
    C_aux: float
       MUSD, capital cost of aux heating

    Outputs
    -------
    C_aa: float
       MUSD/a, Averaged costs of aux heating

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        f_spares : float
            Factor for spares.
        annual_aux_heating_factor : float
            Fraction of the capital cost required for maintainance each year.

    Notes
    -----
    This is a constant factor of the aux heating capital cost.
    The value is from the paragraph before Equation (25) of
    :footcite:t:`sheffield_generic_2016`.
    """
    def initialize(self):
        self.options.declare('cost_params', default=None)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("C_aux",
                       units="MUSD",
                       desc="Capital cost of aux heating")
        self.add_output("C_aa",
                        units="MUSD/a",
                        ref=10,
                        desc="Annualized cost of aux heating")

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

    Equation (18) of :footcite:t:`sheffield_generic_2016`.

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
            Overall fudge factor, normally 1.

    Notes
    -----
    The blanket and divertor are considered as Operating Costs.
    """
    def initialize(self):
        self.options.declare('cost_params', default=None, types=dict)

    def setup(self):
        self.cc = self.options['cost_params']
        #  todo: raise a proper error
        self.add_input("P_t",
                       units="MW",
                       desc="Thermal power handled by b.o.p.")
        self.add_input("Cpc", units="GUSD", desc="Cost of primary coilset")
        self.add_input("Csg", units="GUSD", desc="Cost of shielding and gaps")
        self.add_input("Cst",
                       units="GUSD",
                       desc="Cost of structures & supports for coils")
        self.add_input("C_aux",
                       units="GUSD",
                       desc="Cost of aux heating systems")

        self.add_output("C_ht",
                        units="GUSD",
                        ref=1,
                        desc="Cost of main heat transfer steam system")
        self.add_output("C_FI",
                        units="GUSD",
                        ref=1,
                        desc="Cost of the fusion island")

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

    From Equation (19) of :footcite:t:`sheffield_generic_2016`,

    .. math::
       C_D = f_{cont}\left(\left(c_{e1} + c_{e2} P_e/c_{e3}\right)
                           \left(\frac{P_t}{d_{Pt}}\right)^{e_{Pt}}
                           + c_V\left(\frac{V_{FI}}{d_{V}}\right)^{e_{V}}
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
    cost_params : dict
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
    """
    def initialize(self):
        self.options.declare('cost_params', types=dict)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("P_e",
                       units="MW",
                       desc="Electric power generated by turbine")
        self.add_input("P_t", units="MW", desc="Thermal power of the system")
        self.add_input("V_FI",
                       units="m**3",
                       desc="Volume of the fusion island")
        self.add_input("C_FI",
                       units="GUSD",
                       desc="Cost of fusion island components")

        self.add_output("C_BOP",
                        units="GUSD",
                        ref=1,
                        desc="Cap. cost of the b.o.p. systems")
        self.add_output("C_bld",
                        units="GUSD",
                        ref=1,
                        desc="Cap. cost of buildings, etc")
        self.add_output("C_D",
                        units="GUSD",
                        ref=1,
                        desc="Direct cap. cost of whole plant")

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


class DeuteriumVariableCost(om.ExplicitComponent):
    r"""Instantaneous variable cost for deuterium

    This is **not** directly from the Sheffield papers.

    Up to dimensional constants,

    .. math::
       \mathrm{D}_\mathrm{usage} =
       P_\mathrm{fus} \, (1\;\mathrm{hour})/ ρ_{energy}

    where :math:`ρ_{energy}` is the energy density of deuterium fuel,
    which was precomputed for this function; see the Notes.

    .. math:: C_{\mathrm{D},v} = C_\mathrm{deu} \, \mathrm{D}_\mathrm{usage}

    Inputs
    ------
    P_fus: float
        MW, Direct fusion power (DT, specifically)

    Outputs
    -------
    C_Dv: float
        kUSD/h, Instantaneous deuterium cost
    D usage : float
        g/h, Rate of burning deuterium mass

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        C_deu  : float
            kUSD/kg, Price of Deuterium in $/kg.

    For a similar computation see pg 512 of Volume II of the STARFIRE report
    :footcite:p:`baker_starfire_1980-1`.

    The price of deuterium is estimated from wikipedia: it lists $13k/kg, but I
    estimated it as $10k/kg.

    Notes
    -----
    Mathematica code to compute the constant:

    .. code-block:: Mathematica

        mt = Entity["Isotope", "Hydrogen3"]["AtomicMass"];
        md = Quantity["DeuteronMass"];
        mn = Quantity["NeutronMass"];
        ma = Quantity["AlphaParticleMass"];
        me = Quantity["ElectronMass"];
        c = Quantity["SpeedOfLight"];
        jperkg = UnitConvert[(mt + md - mn - ma - me) c^2/md,
          "Megajoules"/"Kilograms"]
    """
    def initialize(self):
        self.options.declare('cost_params', types=dict)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("P_fus", units="MW", desc="DT fusion power")
        self.add_output("D usage",
                        units="kg/h",
                        ref=0.01,
                        desc="Rate of deuterium burning")
        self.add_output("C_Dv",
                        units="USD/h",
                        ref=0.5,
                        desc="Instantaneous deuterium cost")

        # this was done by looking up the masses of D+, T, α,
        # neutrons, and e⁻, and doing Δm c²
        self.energy_density_d_only = 8.42840e8  # MJ per kg of D

    def compute(self, inputs, outputs):
        cc = self.cc
        P_fus = inputs["P_fus"]
        C_per_kg = cc['C_deu_per_kg']
        outputs["D usage"] = P_fus * hour / self.energy_density_d_only
        outputs["C_Dv"] = C_per_kg * outputs["D usage"]

    def setup_partials(self):
        cc = self.cc
        C_per_kg = cc['C_deu_per_kg']
        dDusage_dPfus = hour / self.energy_density_d_only
        dCdeut_dPfus = C_per_kg * dDusage_dPfus

        self.declare_partials("D usage", ["P_fus"], val=dDusage_dPfus)
        self.declare_partials("C_Dv", ["P_fus"], val=dCdeut_dPfus)


class AnnualDeuteriumCost(om.ExplicitComponent):
    r"""Similar to the STARFIRE report.

    This is **not** directly from the Sheffield papers.

    Assumes a D-T reactor.

    .. math:: C_{deuterium} = f_{av} \, C_{\mathrm{D},v} \,
       \frac{\mathrm{a}}{\mathrm{h}}

    .. math:: \textrm{Annual D usage} = f_{av}\,\mathrm{D}_\mathrm{usage} \,
        \frac{\mathrm{a}}{\mathrm{h}}\, \frac{\mathrm{g}}{\mathrm{kg}}

    Inputs
    ------
    C_Dv: float
       kUSD/h, Instantaneous variable cost of Deuterium
    D usage : float
       g/h
    f_av: float
       Availability factor

    Outputs
    -------
    C_deuterium: float
       kUSD/a, Average annual deuterium cost
    Annual D usage: float
       kg/a

    Notes
    -----
    f_av here is the fraction of time during which the plasma is in flattop.

    For a similar computation see pg 512 of Volume II of the STARFIRE report
    :footcite:p:`baker_starfire_1980-1`.

    The price of deuterium is estimated from wikipedia: it lists $13k/kg, but I
    estimated it as $10k/kg.
    """
    def setup(self):
        self.add_input("C_Dv",
                       units="kUSD/h",
                       desc="Instantaneous variable cost of deuterium")
        self.add_input("D usage",
                       units="g/h",
                       desc="Instantaneous deuterium usage")
        self.add_input("f_av", desc="Availability factor")
        self.add_output("C_deuterium",
                        units="kUSD/a",
                        ref=0.5,
                        desc="Annualized deuterium cost")
        self.add_output("Annual D usage",
                        units="kg/a",
                        ref=50,
                        desc="Annualized deuterium usage")
        self.g_per_h_to_kg_per_a = year / hour / kilo
        self.per_h_to_per_a = year / hour

    def compute(self, inputs, outputs):
        C_Dv = inputs["C_Dv"]
        D_use = inputs["D usage"]
        f_av = inputs["f_av"]
        outputs["C_deuterium"] = f_av * C_Dv * self.per_h_to_per_a
        outputs["Annual D usage"] = f_av * D_use * self.g_per_h_to_kg_per_a

    def setup_partials(self):
        self.declare_partials("C_deuterium", ["C_Dv", "f_av"])
        self.declare_partials("Annual D usage", ["D usage", "f_av"])

    def compute_partials(self, inputs, J):
        C_Dv = inputs["C_Dv"]
        D_use = inputs["D usage"]
        f_av = inputs["f_av"]
        J["C_deuterium", "C_Dv"] = f_av * self.per_h_to_per_a
        J["C_deuterium", "f_av"] = C_Dv * self.per_h_to_per_a
        J["Annual D usage", "D usage"] = f_av * self.g_per_h_to_kg_per_a
        J["Annual D usage", "f_av"] = D_use * self.g_per_h_to_kg_per_a


class MiscReplacements(om.ExplicitComponent):
    r"""Miscellaneous replacements

    .. math::
       C_{misca} &= f_{CR,0} \, C_{misc}

       C_{fa} &= C_{fuel} + C_{misca}

    Inputs
    ------
    C_fuel: float
        MUSD/a, Annual cost of fuel (Deuterium? Lithium?)

    Outputs
    -------
    C_fa : float
        MUSD/a, Cost of misc scheduled replaceable items and fuel
    C_misca : float
        MUSD/a, Cost of misc scheduled replaceable items

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        C_misc: float
            MUSD, Capital cost of miscellaneous scheduled-replaceable items.

    Notes
    -----
    The original paper uses C_fa = 0.4 + 24 f_CR0.
    The new paper :footcite:p:`sheffield_generic_2016` uses an
    inflation factor of 2.19 from 1983 to 2010, but also says C_fa = 7.5 M
    with f_CR0 = 0.078. I'm not sure how this is consistent.

    Here the default C_misc is from multiplying 24 * 2.2.
    """
    def initialize(self):
        self.options.declare('cost_params', types=dict)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("C_fuel",
                       units="MUSD/a",
                       val=0.8,
                       desc="Annual cost of fuel")
        self.add_output("C_fa",
                        units="MUSD/a",
                        ref=10,
                        desc="Cost of misc. sched. repl. items & fuel")
        self.add_output("C_misca",
                        units="MUSD/a",
                        ref=10,
                        desc="Cost of misc. sched. repl. items")

    def compute(self, inputs, outputs):
        cc = self.cc
        f_CR0 = cc['f_CR0']
        C_misc = cc['C_misc']
        C_fuel = inputs['C_fuel']
        C_fa = C_fuel + f_CR0 * C_misc
        outputs['C_fa'] = C_fa
        outputs['C_misca'] = f_CR0 * C_misc

    def setup_partials(self):
        self.declare_partials('C_fa', 'C_fuel', val=1)


class FuelCycleCost(om.ExplicitComponent):
    r"""Fuel cycle cost sum

    .. math:: C_F = C_{ba} + C_{ta} + C_{aa} + C_{fa}

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

    From Equation (22) of :footcite:t:`sheffield_generic_2016`.
    """
    def setup(self):
        self.add_input("C_ba",
                       units="MUSD/a",
                       val=0.0,
                       desc="Annualized blanket costs")
        self.add_input("C_ta",
                       units="MUSD/a",
                       val=0.0,
                       desc="Annualized divertor costs")
        self.add_input("C_aa",
                       units="MUSD/a",
                       val=0.0,
                       desc="Annualized aux heating costs")
        self.add_input("C_fa",
                       units="MUSD/a",
                       val=0.0,
                       desc="Annualized fuel costs")
        self.add_output("C_F",
                        units="GUSD/a",
                        ref=0.100,
                        desc="Annualized 'fuel cycle' costs")

    def compute(self, inputs, outputs):
        outputs["C_F"] = (inputs["C_ba"] + inputs["C_ta"] + inputs["C_aa"] +
                          inputs["C_fa"]) / 1000

    def setup_partials(self):
        self.declare_partials("C_F", ["C_ba", "C_ta", "C_aa", "C_fa"],
                              val=0.001)


class AveragedAnnualBlanketCost(om.ExplicitComponent):
    r"""
    From Equation (23) of :footcite:t:`sheffield_generic_2016`.

    .. math::

        C_{ba} = f_{fail} \left(f_{spare} C_{bl} F_{CR0}
                        + (f_{av} N p_{wn} / F_{wn} - 1)C_{bl}/N\right)

    where :math:`N` is the number of years of plant operation.

    Inputs
    ------
    C_bl : float
        MUSD, cost of the initial blanket
    p_wn : float
        MW/m**2, Neutron wall loading
    F_wn : float
        First wall and blanket lifetime
    f_av : float
        Plant availability
    N_years : float
        Number of years of plant operation

    Outputs
    -------
    C_bv : float
        MUSD/a, Variable cost of blanket usage, instantaneous.
    Initial blanket : float
        MUSD/a: Annualized cost of the initial blanket
    Avg blanket repl : float
        MUSD/a, Averaged annual blanket replacement costs
    C_ba : float
        MUSD/a, Averaged annual blanket costs

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        f_failures : float
            Contigency factor for failures.
        f_spares : float
            Factor for spares.
        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor, normally 1.
    """
    def initialize(self):
        self.options.declare('cost_params', types=dict)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("C_bl",
                       units="MUSD",
                       val=0.0,
                       desc="Cost of 1st blanket set")
        self.add_input("p_wn", units="MW/m**2", desc="Neutron wall loading")
        self.add_input("F_wn", units="MW*a/m**2", desc="Blanket durability")
        self.add_input("f_av", val=1.0, desc="Plant availability factor")
        self.add_input("N_years", desc="Plant operational lifetime")
        self.add_output("C_bv",
                        units="MUSD/a",
                        lower=0,
                        ref=40,
                        desc="Variable cost of blanket")
        self.add_output("Initial blanket",
                        units="MUSD/a",
                        lower=0,
                        ref=20,
                        desc="Annualized cost of 1st blanket")
        self.add_output("Avg blanket repl",
                        units="MUSD/a",
                        lower=0,
                        ref=30,
                        desc="Annualized replacement blanket costs")
        self.add_output("C_ba",
                        units="MUSD/a",
                        lower=0,
                        ref=50,
                        desc="Annualized blanket costs")

        self.ffails = self.cc['fudge'] * self.cc['f_failures']

    def compute(self, inputs, outputs):
        cc = self.cc
        ffails = self.ffails
        C_bl = inputs["C_bl"]
        p_wn = inputs["p_wn"]
        F_wn = inputs["F_wn"]
        f_av = inputs["f_av"]
        n_years = inputs["N_years"]

        C_bv_raw = C_bl * p_wn / F_wn

        outputs["C_bv"] = ffails * C_bv_raw

        initial_bl = cc['f_spares'] * cc['F_CR0'] * C_bl
        outputs["Initial blanket"] = ffails * initial_bl
        # averaged cost of scheduled blanket replacement
        avg_sched_repl = (f_av * n_years * p_wn / F_wn - 1) * C_bl / n_years
        outputs["Avg blanket repl"] = ffails * avg_sched_repl
        outputs["C_ba"] = ffails * (initial_bl + avg_sched_repl)

    def setup_partials(self):
        cc = self.cc
        dinitial_dCbl = cc['F_CR0'] * cc['f_spares'] * self.ffails
        self.declare_partials("C_bv", ["C_bl", "p_wn", "F_wn"])
        self.declare_partials("Initial blanket", ["C_bl"], val=dinitial_dCbl)
        self.declare_partials("Avg blanket repl",
                              ["C_bl", "f_av", "p_wn", "F_wn", "N_years"])
        self.declare_partials("C_ba",
                              ["C_bl", "f_av", "p_wn", "F_wn", "N_years"])

    def compute_partials(self, inputs, J):
        ffails = self.ffails
        C_bl = inputs["C_bl"]
        f_av = inputs["f_av"]
        p_wn = inputs["p_wn"]
        F_wn = inputs["F_wn"]
        n_years = inputs["N_years"]

        J["C_bv", "C_bl"] = ffails * p_wn / F_wn
        J["C_bv", "p_wn"] = ffails * C_bl / F_wn
        J["C_bv", "F_wn"] = -ffails * C_bl * p_wn / F_wn**2

        J["Avg blanket repl",
          "C_bl"] = ffails * (f_av * p_wn / F_wn - 1 / n_years)
        J["Avg blanket repl", "f_av"] = ffails * C_bl * p_wn / F_wn
        J["Avg blanket repl", "p_wn"] = f_av * J["C_bv", "p_wn"]
        J["Avg blanket repl", "F_wn"] = f_av * J["C_bv", "F_wn"]
        J["Avg blanket repl", "N_years"] = ffails * C_bl / n_years**2

        J["C_ba", "C_bl"] = (J["Initial blanket", "C_bl"] +
                             J["Avg blanket repl", "C_bl"])
        J["C_ba", "p_wn"] = J["Avg blanket repl", "p_wn"]
        J["C_ba", "F_wn"] = J["Avg blanket repl", "F_wn"]
        J["C_ba", "f_av"] = J["Avg blanket repl", "f_av"]
        J["C_ba", "N_years"] = J["Avg blanket repl", "N_years"]


class DivertorVariableCost(om.ExplicitComponent):
    r"""Instantaneous variable cost of divertor usage

    .. math::
        C_{tv} = C_tt p_{tt} / F_{tt}

    Inputs
    ------
    C_tt : float
        MUSD, cost of the initial divertor
    p_tt : float
        MW/m**2, Averaged thermal load on the divertor targets
    F_tt : float
        MW*a/m**2, Divertor durability

    Outputs
    -------
    C_tv : float
        MUSD/a, Variable cost of divertor usage, instantaneous.

    Notes
    -----
    This is a subelement of Equation (24) of
    :footcite:t:`sheffield_generic_2016`.
    """
    def setup(self):
        self.add_input("C_tt",
                       units="MUSD",
                       val=0.0,
                       desc="Cost of 1st divertor")
        self.add_input("F_tt", units="MW*a/m**2", desc="Divertor durability")
        self.add_input("p_tt",
                       units="MW/m**2",
                       desc="Thermal load on divertor")
        self.add_output("C_tv",
                        units="MUSD/a",
                        lower=0,
                        ref=100,
                        desc="Variable costs from divertor")

    def compute(self, inputs, outputs):
        C_tt = inputs["C_tt"]
        F_tt = inputs["F_tt"]
        p_tt = inputs["p_tt"]

        outputs["C_tv"] = C_tt * p_tt / F_tt

    def setup_partials(self):
        self.declare_partials("C_tv", ["C_tt", "p_tt", "F_tt"])

    def compute_partials(self, inputs, J):
        C_tt = inputs["C_tt"]
        F_tt = inputs["F_tt"]
        p_tt = inputs["p_tt"]

        J["C_tv", "C_tt"] = p_tt / F_tt
        J["C_tv", "p_tt"] = C_tt / F_tt
        J["C_tv", "F_tt"] = -C_tt * p_tt / F_tt**2


class AveragedAnnualDivertorCost(om.ExplicitComponent):
    r"""
    From Equation (24) of :footcite:t:`sheffield_generic_2016`.

    .. math::
        C_{ta} &= f_{fail} \left(f_{spare} C_{tt} F_{CR0}
                        + (f_{av} N p_{tt} / F_{tt} - 1)C_{tt}/N\right)

               &= f_{fail} \left(f_{spare} C_{tt} F_{CR0}
                        + (f_{av} N C_{tv} - C_{tt})/N\right)

    where :math:`N` is the number of years of plant operation.
    :math:`C_{tv}` is the instantaneous variable cost of divertor usage.

    Inputs
    ------
    C_tt : float
        MUSD, cost of the initial divertor
    p_tt : float
        MW/m**2, Thermal load on divertor targets
    F_tt : float
        Divertor durability
    f_av : float
        Plant availability
    N_years : float
        Number of years of plant operation

    Outputs
    -------
    C_tv : float
        MUSD/a, Variable cost of divertor usage, instantaneous.
    Initial divertor : float
        MUSD/a: Annualized cost of the initial divertor
    Avg divertor repl : float
        MUSD/a, Averaged annual divertor replacement costs
    C_ta : float
        MUSD/a, Averaged annual divertor costs

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        f_failures : float
            Contigency factor for failures.
        f_spares : float
            Factor for spares.
        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor, normally 1.
    """
    def initialize(self):
        self.options.declare('cost_params', types=dict)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("C_tt",
                       units="MUSD",
                       val=0.0,
                       desc="Costs of 1st divertor")
        self.add_input("p_tt",
                       units="MW/m**2",
                       desc="Thermal load on divertor target")
        self.add_input("F_tt", units="MW*a/m**2", desc="Divertor durability")
        self.add_input("f_av", val=1.0, desc="Plant availability factor")
        self.add_input("N_years", desc="Plant operational lifetime")
        self.add_output("C_tv",
                        units="MUSD/a",
                        lower=0,
                        ref=40,
                        desc="Variable cost of divertor use, instantaneous")
        self.add_output("Initial divertor",
                        units="MUSD/a",
                        lower=0,
                        ref=20,
                        desc="Annualized 1st divertor cost")
        self.add_output("Avg divertor repl",
                        units="MUSD/a",
                        lower=0,
                        ref=30,
                        desc="Annualized replacement divertor cost")
        self.add_output("C_ta",
                        units="MUSD/a",
                        lower=0,
                        ref=50,
                        desc="Annualized divertor costs")

        self.ffails = self.cc['fudge'] * self.cc['f_failures']

    def compute(self, inputs, outputs):
        cc = self.cc
        ffails = self.ffails
        C_tt = inputs["C_tt"]
        p_tt = inputs["p_tt"]
        F_tt = inputs["F_tt"]
        f_av = inputs["f_av"]
        n_years = inputs["N_years"]

        C_tv_raw = C_tt * p_tt / F_tt

        outputs["C_tv"] = ffails * C_tv_raw

        initial_tt = cc['f_spares'] * cc['F_CR0'] * C_tt
        outputs["Initial divertor"] = ffails * initial_tt
        # averaged cost of scheduled divertor replacement
        avg_sched_repl = (f_av * n_years * p_tt / F_tt - 1) * C_tt / n_years
        outputs["Avg divertor repl"] = ffails * avg_sched_repl
        outputs["C_ta"] = ffails * (initial_tt + avg_sched_repl)

    def setup_partials(self):
        cc = self.cc
        dinitial_dCtt = cc['F_CR0'] * cc['f_spares'] * self.ffails
        self.declare_partials("C_tv", ["C_tt", "p_tt", "F_tt"])
        self.declare_partials("Initial divertor", ["C_tt"], val=dinitial_dCtt)
        self.declare_partials("Avg divertor repl",
                              ["C_tt", "f_av", "p_tt", "F_tt", "N_years"])
        self.declare_partials("C_ta",
                              ["C_tt", "f_av", "p_tt", "F_tt", "N_years"])

    def compute_partials(self, inputs, J):
        ffails = self.ffails
        C_tt = inputs["C_tt"]
        f_av = inputs["f_av"]
        p_tt = inputs["p_tt"]
        F_tt = inputs["F_tt"]
        n_years = inputs["N_years"]

        J["C_tv", "C_tt"] = ffails * p_tt / F_tt
        J["C_tv", "p_tt"] = ffails * C_tt / F_tt
        J["C_tv", "F_tt"] = -ffails * C_tt * p_tt / F_tt**2

        J["Avg divertor repl",
          "C_tt"] = ffails * (f_av * p_tt / F_tt - 1 / n_years)
        J["Avg divertor repl", "f_av"] = ffails * C_tt * p_tt / F_tt
        J["Avg divertor repl", "p_tt"] = f_av * J["C_tv", "p_tt"]
        J["Avg divertor repl", "F_tt"] = f_av * J["C_tv", "F_tt"]
        J["Avg divertor repl", "N_years"] = ffails * C_tt / n_years**2

        J["C_ta", "C_tt"] = (J["Initial divertor", "C_tt"] +
                             J["Avg divertor repl", "C_tt"])
        J["C_ta", "p_tt"] = J["Avg divertor repl", "p_tt"]
        J["C_ta", "F_tt"] = J["Avg divertor repl", "F_tt"]
        J["C_ta", "f_av"] = J["Avg divertor repl", "f_av"]
        J["C_ta", "N_years"] = J["Avg divertor repl", "N_years"]


class FixedOMCost(om.ExplicitComponent):
    r"""From Sheffield; scaled based off STARFIRE.

    .. math:: C_{OM} = \mathrm{fudge} \cdot \mathrm{base}_{OM}
       \left(\frac{P_e}{\mathrm{base}_{Pe}}\right)^{1/2}

    Inputs
    ------
    P_e : float
        MW, electric power (net or gross, see note)

    Outputs
    -------
    C_OM : float
        MUSD/a, Operations and maintenance cost

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        base_Pe  : float
            MW, Base electric power for the scaling formula
        base_OM  : float
            MUSD/a, Fixed operations and maintenance cost for a plant that
            generates base_Pe of net electric power.
        fudge  : float
            Overall fudge factor, normally 1.

    Notes
    -----
    It appears that this value assumes a steam plant.
    See Equation (F.10) of :footcite:t:`sheffield_cost_1986`.

    It is based on numbers from STARFIRE, a 1200MWe plant.
    This includes staff costs, annual misc. consumables and equipment,
    annual outside support services, annual general and admin costs,
    annual coolant makeup (=0), annual process material (for water and T)
    processing, annual fuel handling costs (=0), and annual miscellaneous costs
    (training, requalification of operators, rent of equipment, travel).

    The two Sheffield papers
    :footcite:t:`sheffield_cost_1986, sheffield_generic_2016`
    appear to scale this cost with the **net** electric
    power, but it would be more appropriate to scale with the **gross** power,
    since that will determine the size of turbines, buildings, cooling, etc.
    """
    def initialize(self):
        self.options.declare("cost_params", types=dict)

    def setup(self):
        self.cc = self.options["cost_params"]
        self.add_input("P_e", units="MW", desc="Electric power")
        self.add_output("C_OM",
                        units="MUSD/a",
                        ref=100,
                        desc="Operations and maintenance cost")

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
    Equation (27) of :footcite:t:`sheffield_generic_2016`.

    .. math:: C_{C0} = f_{CAP,0}\,f_{IND}\,C_D

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
    """
    def setup(self):
        self.add_input("C_D",
                       units="GUSD",
                       desc="Overnight plant capital cost")
        self.add_input("f_CAP0",
                       val=1.0,
                       desc="Constant-dollar capitalization factor")
        self.add_input("f_IND", val=1.0, desc="Indirect charges factor")
        self.add_output("C_C0",
                        units="GUSD",
                        ref=3,
                        desc="Total plant capital cost")

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

    .. math:: f_{IND} = 1 + 0.5 \, T_{constr} / 8

    Equation (28) of :footcite:t:`sheffield_generic_2016`.

    Inputs
    ------
    T_constr : float
        a, Construction time

    Outputs
    -------
    f_IND : float
        Indirect charges factor

    """
    def setup(self):
        self.add_input("T_constr", units="a", desc="Construction time")
        self.add_output("f_IND", desc="Indirect charges factor")

    def compute(self, inputs, outputs):
        T_constr = inputs["T_constr"]
        f_ind = 1 + 0.5 * T_constr / 8
        outputs["f_IND"] = f_ind

    def setup_partials(self):
        self.declare_partials("f_IND", "T_constr", val=0.5 / 8)


class ConstantDollarCapitalizationFactor(om.ExplicitComponent):
    r"""

    Table IV of :footcite:t:`sheffield_generic_2016`.

    .. math:: f_{CAP,0} = 1 + 0.012 \, T_{constr} + 0.003

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
    """
    def setup(self):
        self.add_input("T_constr", units='a', desc="Construction time")
        self.add_output("f_CAP0", desc="Constant-dollar capitalization factor")

    def compute(self, inputs, outputs):
        T = inputs["T_constr"]
        f_cap0 = 1 + 0.012 * T + 0.003
        outputs["f_CAP0"] = f_cap0

    def setup_partials(self):
        self.declare_partials("f_CAP0", "T_constr", val=0.012)


class CostOfElectricity(om.ExplicitComponent):
    r"""
    Equation (26) of :footcite:t:`sheffield_generic_2016`.

    .. math:: COE = \mathrm{fudge} \cdot
       \left(10^6\frac{C_{C0}\,f_{CR,0} + C_F + C_{OM}}
       {P_e \, 8760 \, f_{av}} + (\textrm{waste charge})\right)

    Inputs
    ------
    C_C0 : float
        MUSD, total capital cost
    C_F : float
        MUSD/a, Annual fuel cycle costs
    C_OM : float
        MUSD/a, Operations and maintenance cost
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
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor, normally 1.
        waste_charge : float
            mUSD/kW/h, Cost of waste disposal
    """
    def initialize(self):
        self.options.declare('cost_params', default=None)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input("C_C0",
                       units="MUSD",
                       val=0.0,
                       desc="Total capital cost")
        self.add_input("C_F",
                       units="MUSD/a",
                       val=0.0,
                       desc="Annual fuel cycle cost")
        self.add_input("C_OM",
                       units="MUSD/a",
                       val=0.0,
                       desc="Operations & maint. cost")
        self.add_input("f_av", val=1.0, desc="Availability factor")
        self.add_input("P_e", units="MW", desc="Net electric power output")

        self.add_output("COE",
                        units="mUSD/kW/h",
                        lower=0,
                        ref=100,
                        desc="Cost of electricity")

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
    From Equation (17) of :footcite:t:`sheffield_generic_2016`.
    """
    def setup(self):
        self.add_input("V_pc",
                       units='m**3',
                       val=0.0,
                       desc="Primary coil volume")
        self.add_output("V_st",
                        units='m**3',
                        ref=100,
                        desc="Magnet structure volume")
        self.c = 0.75

    def compute(self, inputs, outputs):
        outputs["V_st"] = self.c * inputs['V_pc']

    def setup_partials(self):
        self.declare_partials('V_st', 'V_pc', val=self.c)


class GeneromakCosting(om.Group):
    r"""Top-level costing group

    Based on :footcite:t:`sheffield_cost_1986, sheffield_generic_2016`.

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
    config : UserConfigurator
        Configuration tree. Required option.

    exact_generomak : bool
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
        self.options.declare('config', default=None, recordable=False)

    def reformat_cost_parameters(self, f):
        r"""Loads cost parameters from configuration tree

        Parameters
        ----------
        f : UserConfigurator.accessor function

        Returns
        -------
        cost_parameters: dict
            Dictionary with the various basic Generomak costing parameters
        """
        cp = {}
        cp["Finance.F_CR0"] = f(["Finance", "F_CR0"])
        cp["Deuterium.cost_per_mass"] = f(
            ["Consumables", "Deuterium", "cost_per_mass"], units="USD/kg")
        cp["PrimaryCoils.cost_per_vol"] = f(
            ["PrimaryCoils", "cost per volume"], units="MUSD/m**3")
        bl = "Blanket"
        cp[bl + ".cost_per_vol"] = f([bl, "cost per volume"],
                                     units="MUSD/m**3")
        cp[bl + ".f_failures"] = f([bl, "f_failures"])
        cp[bl + ".f_spares"] = f([bl, "f_spares"])
        cp[bl + ".fudge"] = f([bl, "fudge"])

        dv = "Divertor"
        cp[dv + ".cost_per_area"] = f([dv, "cost per area"], units="MUSD/m**2")
        cp[dv + ".f_failures"] = f([dv, "f_failures"])
        cp[dv + ".f_spares"] = f([dv, "f_spares"])
        cp[dv + ".fudge"] = f([dv, "fudge"])

        st = "Structure"
        cp[st + ".cost_per_vol"] = f([st, "cost per volume"],
                                     units="MUSD/m**3")
        sg = "ShieldWithGaps"
        cp[sg + ".cost_per_vol"] = f([sg, "cost per volume"],
                                     units="MUSD/m**3")
        aux = "AuxHeating"
        cp[aux + ".cost_per_watt"] = f([aux, "cost per watt"], units="MUSD/MW")
        cp[aux + ".f_spares"] = f([aux, "f_spares"])
        cp[aux + ".ann_maint_fact"] = f([aux, "annual maintenance factor"])

        rp = "ReferencePlant"
        cp[rp + ".d_Pt"] = f([rp, "d_Pt"], units="MW")
        cp[rp + ".e_Pt"] = f([rp, "e_Pt"])
        cp[rp + ".d_Pe"] = f([rp, "d_Pe"], units="MW")

        fi = "FusionIsland"
        cp[fi + ".c_Pt"] = f([fi, "c_Pt"], units="GUSD")
        cp[fi + ".m_pc"] = f([fi, "m_pc"])
        cp[fi + ".m_sg"] = f([fi, "m_sg"])
        cp[fi + ".m_st"] = f([fi, "m_st"])
        cp[fi + ".fudge"] = f([fi, "fudge"])

        cap = "CapitalCost"
        cp[cap + ".contingency"] = f([cap, "contingency"])
        cp[cap + ".c_e1"] = f([cap, "c_e1"], units="GUSD")
        cp[cap + ".c_e2"] = f([cap, "c_e2"], units="GUSD")
        cp[cap + ".c_V"] = f([cap, "c_V"], units="GUSD")
        cp[cap + ".d_V"] = f([cap, "d_V"], units="m**3")
        cp[cap + ".e_V"] = f([cap, "e_V"])
        cp[cap + ".fudge"] = f([cap, "fudge"])

        miscr = "MiscReplacements"
        cp[miscr + ".c_misc"] = f([miscr, "c_misc"], units="MUSD/a")

        fom = "FixedOM"
        cp[fom + ".base_cost"] = f([fom, "base_cost"], units="MUSD/a")
        cp[fom + ".fudge"] = f([fom, "fudge"])
        cp["COE.waste_charge"] = f(["WasteCharge"], units="mUSD/kW/h")
        cp["COE.fudge"] = f(["COE", "fudge"])
        return cp

    def setup(self):
        exact_generomak = self.options['exact_generomak']
        config = self.options['config']
        f = config.accessor(["costing"])
        genx_interface = f(["GenXInterface"])
        f = config.accessor(["costing", "Generomak"])
        cost_pars = self.reformat_cost_parameters(f)

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
                                           V_st={
                                               'units': 'm**3',
                                               'desc': "Structure volume"
                                           }),
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
        self.add_subsystem("d_cost_var",
                           DeuteriumVariableCost(cost_params=deu_cc),
                           promotes_inputs=["P_fus"])
        self.add_subsystem("ann_d_cost",
                           AnnualDeuteriumCost(),
                           promotes_inputs=["f_av"])
        self.connect("d_cost_var.C_Dv", "ann_d_cost.C_Dv")
        self.connect("d_cost_var.D usage", "ann_d_cost.D usage")

        if exact_generomak:
            # The 2016 Generomak paper uses a constant 7.5M for miscellaneous
            # costs.
            ivc = om.IndepVarComp()
            ivc.add_output("C_fa",
                           units="MUSD/a",
                           val=7.5,
                           desc="Annual fuel cycle costs")
            ivc.add_output(
                "C_misca",
                units="MUSD/a",
                val=7.5,  # yes, the same value.
                desc="Annual misc. replacements cost")
            self.add_subsystem("misc",
                               ivc,
                               promotes_outputs=["C_fa", "C_misca"])
        else:
            # This allows a variation due to the price of buying deuterium.
            misc_cc = {
                'f_CR0': cost_pars["Finance.F_CR0"],
                'C_misc': cost_pars["MiscReplacements.c_misc"]
            }
            self.add_subsystem("misc",
                               MiscReplacements(cost_params=misc_cc),
                               promotes_outputs=["C_misca", "C_fa"])
            self.connect("ann_d_cost.C_deuterium", "misc.C_fuel")

        ann_aux_cc = {
            'f_spares': cost_pars["AuxHeating.f_spares"],
            'annual_aux_heating_factor': cost_pars["AuxHeating.ann_maint_fact"]
        }
        self.add_subsystem("annual_aux",
                           AnnualAuxHeatingCost(cost_params=ann_aux_cc),
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
            FusionIslandCost(cost_params=fi_cc),
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
                           CapitalCost(cost_params=cap_cc),
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
            AveragedAnnualBlanketCost(cost_params=ann_bl_cc),
            promotes_inputs=["f_av", "N_years", "C_bl", "p_wn", "F_wn"],
            promotes_outputs=["C_ba"])

        ann_dv_cc = {
            'f_failures': cost_pars["Divertor.f_failures"],
            'f_spares': cost_pars["Divertor.f_spares"],
            'F_CR0': cost_pars["Finance.F_CR0"],
            'fudge': cost_pars["Divertor.fudge"],
        }
        self.add_subsystem(
            "ann_divertor_cost",
            AveragedAnnualDivertorCost(cost_params=ann_dv_cc),
            promotes_inputs=["f_av", "N_years", "C_tt", "F_tt", "p_tt"],
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
                           FixedOMCost(cost_params=fom_cc),
                           promotes_outputs=["C_OM"])

        if exact_generomak:
            # use the _net_ electric generation only. This ignores the cost of
            # capital to generate the recirculating power.
            self.promotes("plant_capital", inputs=[("P_e", "P_net")])
            self.promotes("omcost", inputs=[("P_e", "P_net")])
            # create a stub so that P_e has something to connect to.
            self.add_subsystem("ignore2",
                               om.ExecComp(["ignore=P_e"],
                                           P_e={
                                               'units': 'MW',
                                               'desc': "Gross electric power"
                                           }),
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
            CostOfElectricity(cost_params=coe_cc),
            promotes_inputs=["C_C0", "C_F", ("P_e", "P_net"), "f_av", "C_OM"],
            promotes_outputs=["COE"])

        if genx_interface:
            self.add_subsystem("GenXInterface",
                               GeneromakToGenX(cost_params=coe_cc),
                               promotes_inputs=[
                                   "C_C0", ("P_e", "P_net"), "C_aa", "C_misca",
                                   "C_OM"
                               ])
            self.connect("ann_blanket_cost.C_bv", "GenXInterface.C_bv")
            self.connect("ann_blanket_cost.Initial blanket",
                         "GenXInterface.Initial blanket")
            self.connect("ann_divertor_cost.C_tv", "GenXInterface.C_tv")
            self.connect("ann_divertor_cost.Initial divertor",
                         "GenXInterface.Initial divertor")
            # in the exact formulation, C_misca contains the fuel cost
            if not exact_generomak:
                self.connect("d_cost_var.C_Dv", "GenXInterface.C_fv")


#################################################
# Interface from Generomak costing to GenX inputs


class GeneromakToGenX(om.ExplicitComponent):
    r"""Outputs metrics used by GenX

    This is an interface from the Sheffield costing model
    to the cost-related inputs required by a standard GenX generator.

    For GenX, the 'fuel' used for the reactor will be 'None'; instead,
    the deuterium is included in the variable cost of operation.
    This avoids the need for an additional fuel type in GenX.

    Inputs
    ------

    C_C0 : float
        MUSD, total capital cost
    Initial blanket : float
        MUSD/a, annualized cost of the initial blanket
    Initial divertor : float
        MUSD/a, annualized cost of the initial divertor

    C_aa: float
       MUSD/a, Averaged costs of aux heating
    C_misca : float
       MUSD/a, Miscellaneous replacements
    C_OM : float
       MUSD/a, 'other' fixed operations and maintenance

    C_bv : float
        USD/h, variable cost of blanket usage
    C_tv : float
        USD/h, variable cost of divertor usage
    C_fv : float
        USD/h, variable cost of fuel

    P_e : float
        MW, Electric power output

    Outputs
    -------
    Inv_Cost_per_MWyr : float
        MUSD/MW/a, Annualized investment cost per MW.
    Fixed_OM_Cost_per_MWyr : float
        MUSD/MW/a, Operations and maintenance cost
    Var_OM_Cost_per_MWh : float
        USD/MW/h, Instantaneous variable cost, not including downtime
        for repairs.

    Notes
    -----
    The three outputs use exact GenX nomenclature.

    Options
    -------
    cost_params : dict
        This is a dictionary of coefficients for the costing model.
        The dictionary provided must have values for all these keys:

        F_CR0  : float
            1/a, Constant dollar fixed charge rate.
        fudge  : float
            Overall fudge factor, normally 1.
        waste_charge : float
            mUSD/kW/h, Cost of waste disposal
    """
    def initialize(self):
        self.options.declare('cost_params', default=None)

    def setup(self):
        self.cc = self.options['cost_params']
        self.add_input('C_C0', units='USD', val=0.0, desc="Total capital cost")
        self.add_input('Initial blanket',
                       units='USD/a',
                       val=0.0,
                       desc="Annualized cost of 1st blanket")
        self.add_input('Initial divertor',
                       units='USD/a',
                       val=0.0,
                       desc="Annualized cost of 1st divertor")

        self.add_input('C_aa',
                       units='USD/a',
                       val=0.0,
                       desc="Annualized cost of aux heating")
        self.add_input('C_misca',
                       units='USD/a',
                       val=0.0,
                       desc="Miscallaneous replacements cost")
        self.add_input('C_OM',
                       units='USD/a',
                       val=0.0,
                       desc="Operations & maint. cost")

        self.add_input('C_bv',
                       units='USD/h',
                       val=0.0,
                       desc="Variable ops. & maint. cost of blanket")
        self.add_input('C_tv',
                       units='USD/h',
                       val=0.0,
                       desc="Variable ops. & maint. cost of divertor")
        self.add_input('C_fv',
                       units='USD/h',
                       val=0.0,
                       desc="Variable ops. & maint. cost of fuel")

        self.add_input('P_e', units='MW', desc="Electric power output")

        self.add_output('Inv_Cost_per_MWyr',
                        units='USD/MW/a',
                        ref=100000,
                        desc="Annualized investment cost per MW")
        self.add_output('Fixed_OM_Cost_per_MWyr',
                        units='USD/MW/a',
                        lower=0,
                        ref=10000,
                        desc="Fixed operations and maintenance cost")
        self.add_output('Var_OM_Cost_per_MWh',
                        units='USD/MW/h',
                        lower=0,
                        ref=10,
                        desc="Instantaneous variable ops. & maint.  cost")

    def compute(self, inputs, outputs):
        cc = self.cc
        f_cr0 = cc['F_CR0']
        fudge = cc['fudge']

        c_c0 = inputs['C_C0']
        initial_blanket = inputs['Initial blanket']
        initial_divertor = inputs['Initial divertor']

        c_aa = inputs['C_aa']
        c_misca = inputs['C_misca']
        c_om = inputs['C_OM']

        c_bv = inputs['C_bv']
        c_tv = inputs['C_tv']
        c_fv = inputs['C_fv']

        pe = inputs['P_e']

        inv = (c_c0 * f_cr0 + initial_blanket + initial_divertor) / pe
        fom = (c_aa + c_misca + c_om) / pe
        vom = (c_bv + c_tv + c_fv) / pe + cc['waste_charge']

        outputs['Inv_Cost_per_MWyr'] = fudge * inv
        outputs['Fixed_OM_Cost_per_MWyr'] = fudge * fom
        outputs['Var_OM_Cost_per_MWh'] = fudge * vom

    def setup_partials(self):
        self.declare_partials(
            'Inv_Cost_per_MWyr',
            ['C_C0', 'Initial blanket', 'Initial divertor', 'P_e'])
        self.declare_partials('Fixed_OM_Cost_per_MWyr',
                              ['C_aa', 'C_misca', 'C_OM', 'P_e'])
        self.declare_partials('Var_OM_Cost_per_MWh',
                              ['C_bv', 'C_tv', 'C_fv', 'P_e'])

    def compute_partials(self, inputs, J):
        cc = self.cc
        c_c0 = inputs['C_C0']
        bla1 = inputs['Initial blanket']
        div1 = inputs['Initial divertor']

        c_aa = inputs['C_aa']
        c_misca = inputs['C_misca']
        c_om = inputs['C_OM']

        c_bv = inputs['C_bv']
        c_tv = inputs['C_tv']
        c_fv = inputs['C_fv']

        pe = inputs['P_e']
        fudge = cc['fudge']
        f_cr0 = cc['F_CR0']

        J['Inv_Cost_per_MWyr', 'C_C0'] = fudge * f_cr0 / pe
        J['Inv_Cost_per_MWyr', 'Initial blanket'] = fudge / pe
        J['Inv_Cost_per_MWyr', 'Initial divertor'] = fudge / pe
        J['Inv_Cost_per_MWyr',
          'P_e'] = -fudge * (c_c0 * f_cr0 + bla1 + div1) / pe**2

        J['Fixed_OM_Cost_per_MWyr', 'C_aa'] = fudge / pe
        J['Fixed_OM_Cost_per_MWyr', 'C_misca'] = fudge / pe
        J['Fixed_OM_Cost_per_MWyr', 'C_OM'] = fudge / pe
        J['Fixed_OM_Cost_per_MWyr',
          'P_e'] = -fudge * (c_aa + c_misca + c_om) / pe**2

        J['Var_OM_Cost_per_MWh', 'C_bv'] = fudge / pe
        J['Var_OM_Cost_per_MWh', 'C_tv'] = fudge / pe
        J['Var_OM_Cost_per_MWh', 'C_fv'] = fudge / pe
        J['Var_OM_Cost_per_MWh', 'P_e'] = -fudge * (c_bv + c_tv + c_fv) / pe**2


if __name__ == '__main__':
    prob = om.Problem()
    uc = UserConfigurator()

    prob.model = GeneromakCosting(exact_generomak=True, config=uc)

    prob.setup(force_alloc_complex=True)

    # geometry
    prob.set_val("V_FI", 5152, units='m**3')
    prob.set_val("V_pc", 607, units='m**3')
    prob.set_val("V_sg", 914, units='m**3')
    prob.set_val("V_st", 607 * 0.75, units='m**3')
    prob.set_val("V_bl", 321, units='m**3')
    prob.set_val("A_tt", 60, units='m**2')

    # power levels
    prob.set_val("P_aux", 50, units='MW')
    prob.set_val("P_fus", 2250, units="MW")
    prob.set_val("P_t", 2570, units="MW")
    prob.set_val("P_e", 1157, units="MW")
    prob.set_val("P_net", 1000, units="MW")

    # blanket and divertor power fluxes
    prob.set_val("p_wn", 3, units='MW/m**2')
    prob.set_val("p_tt", 10, units="MW/m**2")
    prob.set_val("F_wn", 15, units="MW*a/m**2")
    prob.set_val("F_tt", 10, units="MW*a/m**2")

    # fraction of year with fusion
    prob.set_val("f_av", 0.8)

    # plant lifetime
    prob.set_val("N_years", 30)

    # construction time
    prob.set_val("T_constr", 6, units='a')

    prob.run_driver()
    prob.model.list_inputs(units=True, desc=True)
    prob.model.list_outputs(units=True, desc=True)
