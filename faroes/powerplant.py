from faroes.configurator import UserConfigurator
from faroes.coolantpumping import SimpleCoolantPumpingPower
from faroes.plasmacontrolsystem import SimplePlasmaControlPower
import openmdao.api as om


class AuxilliaryPower(om.ExplicitComponent):
    r"""
    Inputs
    ------
    P_NBI : float
        MW, Injected neutral beam power
    η_NBI : float
        Wall-plug efficiency of neutral-beam heating
    P_RF : float
        MW, RF power that heats the plasma
    η_RF : float
        Wall-plug efficiency of RF heating

    Outputs
    -------
    P_aux,h : float
       MW, total auxilliary heating power to plasma
    P_aux,e : float
       MW, wall-plug aux heating/current drive power
    """
    def setup(self):
        self.add_input("P_NBI",
                       units="MW",
                       val=0,
                       desc="Injected neutral beam power")
        self.add_input("P_RF",
                       units="MW",
                       val=0,
                       desc="Injected radiofrequency heating power")
        self.add_input("η_NBI",
                       val=1,
                       desc="Wall-plug efficiency of neutral-beam heating")
        self.add_input("η_RF",
                       val=1,
                       desc="Wall-plug efficiency of RF heating")
        self.add_output("P_aux,h",
                        units="MW",
                        desc="Total auxilliary heating power to plasma")
        self.add_output("P_aux,e", units="MW", desc="Total wall-plug power")

    def compute(self, inputs, outputs):
        P_NBI = inputs["P_NBI"]
        P_RF = inputs["P_RF"]
        η_NBI = inputs["η_NBI"]
        η_RF = inputs["η_RF"]
        P_aux = P_NBI + P_RF
        P_auxe = P_NBI / η_NBI + P_RF / η_RF
        outputs["P_aux,h"] = P_aux
        outputs["P_aux,e"] = P_auxe

    def setup_partials(self):
        self.declare_partials(["P_aux,h", "P_aux,e"], ["P_NBI", "P_RF"])
        self.declare_partials(["P_aux,e"], ["η_NBI", "η_RF"])

    def compute_partials(self, inputs, J):
        P_NBI = inputs["P_NBI"]
        P_RF = inputs["P_RF"]
        η_NBI = inputs["η_NBI"]
        η_RF = inputs["η_RF"]
        J["P_aux,h", "P_NBI"] = 1
        J["P_aux,h", "P_RF"] = 1
        J["P_aux,e", "P_NBI"] = 1 / η_NBI
        J["P_aux,e", "P_RF"] = 1 / η_RF
        J["P_aux,e", "η_NBI"] = -P_NBI / η_NBI**2
        J["P_aux,e", "η_RF"] = -P_RF / η_RF**2


class RecirculatingElectricPower(om.ExplicitComponent):
    r"""

    Inputs
    ------
    P_aux : float
        MW, Auxilliary heating electric power
    P_cryo : float
        MW, Electric power to drive cryogenic systems
    P_pumps : float
        MW, Electric power to drive pumps (primary cooling)
    P_control : float
        MW, Electric power to drive plasma control systems

    Outputs
    -------
    P_recirc : float
        MW, Recirculating electric power
    """
    def setup(self):
        self.add_input("P_aux",
                       units="MW",
                       val=0,
                       desc="Auxilliary heating electric power")
        self.add_input("P_cryo",
                       units="MW",
                       val=0,
                       desc="Electric power to drive cryogenic systems")
        self.add_input("P_pumps",
                       units="MW",
                       val=0,
                       desc="Electric power to drive pumps (primary cooling)")
        self.add_input("P_control",
                       units="MW",
                       val=0,
                       desc="Electric power to drive plasma control systems")
        self.add_output("P_recirc", units="MW", lower=0)

    def compute(self, inputs, outputs):
        P_aux = inputs["P_aux"]
        P_cryo = inputs["P_cryo"]
        P_pumps = inputs["P_pumps"]
        P_control = inputs["P_control"]
        P_recirc = P_aux + P_cryo + P_pumps + P_control
        outputs["P_recirc"] = P_recirc

    def setup_partials(self):
        self.declare_partials("P_recirc",
                              ["P_aux", "P_cryo", "P_pumps", "P_control"],
                              val=1)


class TotalThermalPower(om.ExplicitComponent):
    r"""
    Inputs
    ------
    P_blanket : float
        MW, Power removed from blanket
    P_α : float
        MW, Total alpha power
    P_aux : float
        MW, Auxilliary heating power to plasma
    P_coolant : float
        MW, Power from pumps dissipated thermally
             in the primary (or secondary...) coolant

    Outputs
    -------
    P_primary_heat : float
        MW, Thermal power that needs to be removed from the tokamak
    P_heat : float
        MW, Thermal power delivered to electricity generating systems
    """
    def setup(self):
        self.add_input("P_blanket",
                       units="MW",
                       val=0,
                       desc="Power removed from the blanket")
        self.add_input("P_α", units="MW", val=0, desc="Total alpha power")
        self.add_input("P_aux",
                       units="MW",
                       val=0,
                       desc="Auxilliary heating power to plasma")
        self.add_input(
            "P_coolant",
            units="MW",
            val=0,
            desc="Power from pumps dissipated thermally in the coolant")
        P_ref = 100
        self.add_output(
            "P_primary_heat",
            units="MW",
            lower=0,
            ref=P_ref,
            desc="Thermal power that needs to be removed from the tokamak")
        self.add_output(
            "P_heat",
            units="MW",
            lower=0,
            ref=P_ref,
            desc="Thermal power delivered to electricity generating systems")

    def compute(self, inputs, outputs):
        P_blanket = inputs["P_blanket"]
        P_α = inputs["P_α"]
        P_aux = inputs["P_aux"]
        P_coolant = inputs["P_coolant"]
        P_primary_heat = P_blanket + P_α + P_aux
        P_heat = P_primary_heat + P_coolant
        outputs["P_primary_heat"] = P_primary_heat
        outputs["P_heat"] = P_heat

    def setup_partials(self):
        self.declare_partials(["P_heat", "P_primary_heat"],
                              ["P_blanket", "P_α", "P_aux"],
                              val=1)
        self.declare_partials(["P_heat"], ["P_coolant"], val=1)


class SimpleGeneratedPower(om.ExplicitComponent):
    r"""Simplified power generation

    .. math::
       P_\mathrm{el} = \eta_{th} * P_\mathrm{heat}

    Inputs
    ------
    P_heat : float
        MW, Power to generators

    Outputs
    -------
    P_el : float
        MW, Electrical power generated
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        self.add_input("P_heat")
        self.add_output("P_el")

        if self.options['config'] is not None:
            self.config = self.options['config']
            ac = self.config.accessor(['machine'])
            self.c = ac(["electrical generation", "efficiency"])
        else:
            self.c = 0.45

    def compute(self, inputs, outputs):
        P_heat = inputs["P_heat"]
        outputs["P_el"] = self.c * P_heat

    def setup_partials(self):
        c = self.c
        self.declare_partials("P_el", ["P_heat"], val=c)


class PowerplantQ(om.ExplicitComponent):
    r"""End power plant efficiency metrics

    .. math::
        Q_\mathrm{eng} = P_\mathrm{gen} / P_\mathrm{recirc}

    .. math::
        P_\mathrm{net} = P_\mathrm{gen} - P_\mathrm{recirc}

    .. math::
        f_\mathrm{recirc} = Q_\mathrm{eng}^{-1}

    Inputs
    ------
    P_gen : float
        MW, Power generated
    P_recirc : float
        MW, Recirculating electric power

    Outputs
    -------
    Q_eng : float
        Engineering Q of the plant
    P_net : float
        MW, net electric power
    f_recirc : float
        Fraction of generated electric power which is recirculated
    """
    def setup(self):
        self.add_input("P_gen", units="MW", desc="Power generated")
        self.add_input("P_recirc",
                       units="MW",
                       desc="Recirculating electric power")
        self.add_output("Q_eng",
                        lower=0,
                        ref=10,
                        desc="Engineering Q of the plant")
        self.add_output("P_net",
                        units="MW",
                        desc="Net electric power generated")
        self.add_output("f_recirc",
                        lower=0,
                        upper=1,
                        ref=0.1,
                        desc="Recirculating power fraction")

    def compute(self, inputs, outputs):
        P_gen = inputs["P_gen"]
        P_recirc = inputs["P_recirc"]
        outputs["Q_eng"] = P_gen / P_recirc
        outputs["f_recirc"] = P_recirc / P_gen
        outputs["P_net"] = P_gen - P_recirc

    def setup_partials(self):
        self.declare_partials(["Q_eng", "f_recirc"], ["P_gen", "P_recirc"])
        self.declare_partials("P_net", "P_gen", val=1)
        self.declare_partials("P_net", "P_recirc", val=-1)

    def compute_partials(self, inputs, J):
        P_gen = inputs["P_gen"]
        P_recirc = inputs["P_recirc"]
        J["Q_eng", "P_gen"] = 1 / P_recirc
        J["Q_eng", "P_recirc"] = -P_gen / P_recirc**2
        J["f_recirc", "P_gen"] = -P_recirc / P_gen**2
        J["f_recirc", "P_recirc"] = 1 / P_gen


class Powerplant(om.Group):
    r"""

    Inputs
    ------
    P_NBI : float
        MW, Injected neutral beam power
    η_NBI : float
        Wall-plug efficiency of neutral-beam heating
    P_RF : float
        MW, RF power that heats the plasma
    η_RF : float
        Wall-plug efficiency of RF heating
    P_aux : float
        MW, Auxilliary heating electric power
    P_cryo : float
        MW, Electric power to drive cryogenic systems

    Outputs
    -------
    P_aux : float
       MW, total auxilliary heating power to plasma
    P_aux,e : float
       MW, wall-plug aux heating/current drive power
    P_recirc : float
        MW, Recirculating electric power
    P_gen : float
        MW, Power generated
    Q_eng : float
        Engineering Q of the plant
    P_net : float
        MW, net electric power
    f_recirc : float
        Fraction of generated electric power which is recirculated
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem(
            "aux",
            AuxilliaryPower(),
            promotes_inputs=["P_NBI", "η_NBI", "P_RF", "η_RF"],
        )
        self.add_subsystem("pumps",
                           SimpleCoolantPumpingPower(config=config),
                           promotes_outputs=["P_pumps"])
        self.add_subsystem("control",
                           SimplePlasmaControlPower(config=config),
                           promotes_outputs=["P_control"])
        self.add_subsystem("recirc_el",
                           RecirculatingElectricPower(),
                           promotes_inputs=["P_cryo", "P_pumps", "P_control"])
        self.connect("aux.P_aux,e", ["recirc_el.P_aux"])
        self.add_subsystem(
            "thermal",
            TotalThermalPower(),
            promotes_inputs=["P_blanket", "P_α"],
        )
        self.connect("aux.P_aux,h", ["thermal.P_aux"])
        self.connect("thermal.P_primary_heat",
                     ["pumps.P_thermal", "control.P_thermal"])
        self.connect("P_pumps", "thermal.P_coolant")

        self.add_subsystem("gen", SimpleGeneratedPower(config=config))
        self.connect("thermal.P_heat", "gen.P_heat")
        self.add_subsystem("overall", PowerplantQ())
        self.connect("gen.P_el", "overall.P_gen")
        self.connect("recirc_el.P_recirc", "overall.P_recirc")


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()

    prob.model = Powerplant(config=uc)

    prob.setup()

    prob.set_val("P_NBI", 10, units="MW")
    prob.set_val("η_NBI", 0.3, units=None)
    prob.set_val("P_α", 25, units="MW")
    prob.set_val("P_blanket", 100, units="MW")

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    # prob.model.nonlinear_solver = om.NonlinearBlockGS()
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.linear_solver = om.DirectSolver()

    prob.run_driver()
    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
