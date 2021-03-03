# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

# from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
# from faroes.radialbuild import MenardSTRadialBuild
from faroes.simple_plasma import ZeroDPlasma
from faroes.nbisource import SimpleNBISource
from faroes.fastparticleslowing import FastParticleSlowing
from faroes.fusionreaction import NBIBeamTargetFusion, TotalDTFusionRate
from faroes.fusionreaction import SimpleFusionAlphaSource

from faroes.confinementtime import ConfinementTime
from faroes.radiation import SimpleRadiation

from faroes.plasma_beta import SpecifiedPressure

from faroes.nbicd import CurrentDriveEfficiency, NBICurrent
from faroes.bootstrap import BootstrapCurrent

from faroes.current import CurrentAndSafetyFactor

import openmdao.api as om


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("plasmageom",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0"],
                           promotes_outputs=["ε", "κa"])

        self.add_subsystem("confinementtime",
                           ConfinementTime(config=config, scaling="default"),
                           promotes_inputs=[("R", "R0"), "ε", "κa", "Bt",
                                            ("n19", "<n_e>"), "Ip"])

        self.add_subsystem("ZeroDPlasma",
                           ZeroDPlasma(config=config),
                           promotes_inputs=["<n_e>"],
                           promotes_outputs=["<T_e>"])

        self.connect("plasmageom.V", ["ZeroDPlasma.V"])
        self.connect("confinementtime.τe", ["ZeroDPlasma.τ_th"])

        # back-connections
        self.connect("ZeroDPlasma.A", ["confinementtime.M"])

        # neutral beam heating
        self.add_subsystem("NBIsource", SimpleNBISource(config=config))
        self.add_subsystem("NBIslowing",
                           FastParticleSlowing(config=config),
                           promotes_inputs=[("ne", "<n_e>"), ("Te", "<T_e>")])
        self.connect("ZeroDPlasma.ni",
                     ["NBIslowing.ni", "alphaslowing.ni", "nbicdEff.ni"])
        self.connect("ZeroDPlasma.Ai",
                     ["NBIslowing.Ai", "alphaslowing.Ai", "nbicdEff.Ai"])
        self.connect("ZeroDPlasma.Zi",
                     ["NBIslowing.Zi", "alphaslowing.Zi", "nbicdEff.Zi"])

        # back-connections
        self.connect("NBIslowing.Wfast", ["ZeroDPlasma.W_fast_NBI"])

        self.connect("NBIsource.S", ["NBIslowing.S", "NBIcurr.S"])
        self.connect("NBIsource.E",
                     ["NBIslowing.Wt", "nbicdEff.Eb", "NBIcurr.Eb"])
        self.connect("NBIsource.A", ["NBIslowing.At", "nbicdEff.Ab"])
        self.connect("NBIsource.Z", ["NBIslowing.Zt", "nbicdEff.Zb"])

        self.add_subsystem("NBIfusion",
                           NBIBeamTargetFusion(),
                           promotes_inputs=["<T_e>"])

        self.connect("NBIsource.P", ["NBIfusion.P_NBI"])

        self.add_subsystem("DTfusion", TotalDTFusionRate())
        self.connect("ZeroDPlasma.rate_fus_th", ["DTfusion.rate_th"])
        self.connect("ZeroDPlasma.P_fus_th", ["DTfusion.P_fus_th"])
        self.connect("NBIfusion.rate_fus", ["DTfusion.rate_NBI"])
        self.connect("NBIfusion.P_fus", ["DTfusion.P_fus_NBI"])

        self.add_subsystem("alphasource", SimpleFusionAlphaSource())

        self.add_subsystem("alphaslowing",
                           FastParticleSlowing(config=config),
                           promotes_inputs=[("ne", "<n_e>"), ("Te", "<T_e>")])
        self.connect("DTfusion.rate_fus", ["alphaslowing.S"])
        self.connect("alphasource.E", ["alphaslowing.Wt"])
        self.connect("alphasource.A", ["alphaslowing.At"])
        self.connect("alphasource.Z", ["alphaslowing.Zt"])

        # back-connections
        self.connect("alphaslowing.Wfast", ["ZeroDPlasma.W_fast_α"])

        self.add_subsystem(
            "P_heat",
            om.ExecComp("P_heat = P_alpha + P_NBI",
                        P_heat={'units': 'MW'},
                        P_alpha={'units': 'MW'},
                        P_NBI={'units': 'MW'}))
        self.connect("DTfusion.P_α", "P_heat.P_alpha")
        self.connect("NBIsource.P", "P_heat.P_NBI")

        self.add_subsystem("radiation", SimpleRadiation(config=config))
        self.connect("P_heat.P_heat", "radiation.P_heat")

        # back-connection
        self.connect("radiation.P_loss",
                     ["ZeroDPlasma.P_loss", "confinementtime.PL"])

        self.add_subsystem("specP",
                           SpecifiedPressure(config=config),
                           promotes_inputs=["Bt", "Ip"])

        self.connect("plasmageom.a", ["specP.a"])
        self.connect("plasmageom.L_pol", ["specP.L_pol"])

        # balance the specified pressure and the actual pressure by changing H
        Hbal = om.BalanceComp()
        Hbal.add_balance('H', normalize=True, eq_units="kPa")
        self.add_subsystem("Hbalance", subsys=Hbal)
        self.connect("Hbalance.H", "confinementtime.H")
        self.connect("specP.p_avg.<p_tot>", "Hbalance.lhs:H")
        self.connect("ZeroDPlasma.<p_tot>", "Hbalance.rhs:H")

        # compute neutral beam current drive
        self.add_subsystem(
            "nbicdEff",
            CurrentDriveEfficiency(config=config),
            promotes_inputs=["R0", ("ne", "<n_e>"), "<T_e>", "ε"])

        self.connect("ZeroDPlasma.Z_eff", ["nbicdEff.Z_eff"])
        self.connect("ZeroDPlasma.vth_e", ["nbicdEff.vth_e"])

        self.connect("NBIslowing.slowingt.ts", ["nbicdEff.τs"])
        self.connect("NBIsource.v", ["nbicdEff.vb"])

        self.add_subsystem('NBIcurr', NBICurrent(config=config))
        self.connect("nbicdEff.It/P", "NBIcurr.It/P")

        # compute bootstrap current
        self.add_subsystem("bootstrap",
                           BootstrapCurrent(),
                           promotes_inputs=["ε", "Ip"])
        self.connect("ZeroDPlasma.thermal pressure fraction",
                     "bootstrap.thermal pressure fraction")
        self.connect("specP.βp", "bootstrap.βp")

        self.add_subsystem('current',
                           CurrentAndSafetyFactor(config=config),
                           promotes_inputs=["R0", "Bt"],
                           promotes_outputs=["Ip", ("n_bar", "<n_e>")])
        self.connect("plasmageom.L_pol_simple", "current.L_pol")
        self.connect("plasmageom.a", "current.a")
        self.connect("NBIcurr.I_NBI", "current.I_NBI")
        self.connect("bootstrap.I_BS", "current.I_BS")
        # back-connections
        self.connect("current.q_star", "bootstrap.q_star")
        self.connect("current.q_min", "bootstrap.q_min")

if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.setup()
    # prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('R0', 3, units='m')
    A = 1.6
    prob.set_val('plasmageom.A', A)
    prob.set_val('specP.A', A)
    prob.set_val('nbicdEff.A', A)
    prob.set_val('Bt', 2.094, units='T')

    # plasma inputs

    # initial inputs
    prob.set_val('Hbalance.H', 1.7)
    prob.set_val('Ip', 14.67, units="MA")
    prob.set_val("<n_e>", 1.26, units="n20")

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.linear_solver = om.DirectSolver()

    # initial values
    prob.set_val("radiation.rad.P_loss", 91, units="MW")

    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=True,
                                          units=True)
