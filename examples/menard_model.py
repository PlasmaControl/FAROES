# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.radialbuild import MenardSTRadialBuild
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

from faroes.blanket import MenardSTBlanketAndShieldGeometry
from faroes.blanket import MenardSTBlanketAndShieldMagnetProtection
from faroes.blanket import MagnetCryoCoolingPower
from faroes.blanket import SimpleBlanketPower
from faroes.blanket import InboardMidplaneNeutronFluxFromRing
from faroes.blanket import NeutronWallLoading
from faroes.blanket import MenardMagnetLifetime

from faroes.sol import SOLAndDivertor

from faroes.powerplant import Powerplant

import openmdao.api as om


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("plasmageom",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", ("A", "aspect ratio")],
                           promotes_outputs=["ε", "κa"])

        self.add_subsystem("radial_build",
                           MenardSTRadialBuild(config=config),
                           promotes_inputs=['CS R_max'])

        self.add_subsystem("magnets",
                           MagnetRadialBuild(config=config),
                           promotes_inputs=["R0"],
                           promotes_outputs=[("B0", "Bt")],
                           )
        self.connect('plasmageom.R_max', ['radial_build.plasma R_max'])
        self.connect('plasmageom.R_min', ['radial_build.plasma R_min'])

        self.connect('radial_build.Ob TF R_min', ['magnets.r_iu'])
        self.connect('radial_build.Ib TF R_min', ['magnets.r_is'])
        self.connect('radial_build.Ib TF R_max', ['magnets.r_ot'])

        self.connect('magnets.Ob TF R_out', ['radial_build.Ob TF R_out'])

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
                           promotes_inputs=["Bt", "Ip", ("A", "aspect ratio")])

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
            promotes_inputs=[("A", "aspect ratio"), "R0", ("ne", "<n_e>"), "<T_e>", "ε"])

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

        # magnet heating model;
        # total power in blanket
        self.add_subsystem("magcryo", MagnetCryoCoolingPower(config=config))
        self.add_subsystem("blanket_P", SimpleBlanketPower(config=config))
        self.connect("DTfusion.P_n", ["magcryo.P_n", "blanket_P.P_n"])

        # powerplant model
        self.add_subsystem("pplant", Powerplant(config=config))
        self.connect("NBIsource.P", ["pplant.P_NBI"])
        self.connect("NBIsource.eff", ["pplant.η_NBI"])
        self.connect("DTfusion.P_α", ["pplant.P_α"])
        self.connect("magcryo.P_c,el", ["pplant.P_cryo"])
        self.connect("blanket_P.P_th", ["pplant.P_blanket"])

        # SOL and divertor model. Useful for constraints.
        self.add_subsystem("SOL",
                           SOLAndDivertor(config=config),
                           promotes_inputs=["R0", "Bt", "Ip"])
        self.connect("plasmageom.a", "SOL.a")
        self.connect("plasmageom.κ", "SOL.κ")
        self.connect("P_heat.P_heat", "SOL.P_heat")

        # Neutron wall flux models. Useful for constraints.
        self.add_subsystem("q_n_IB", InboardMidplaneNeutronFluxFromRing(),
                           promotes_inputs=["R0"])
        self.add_subsystem("q_n", NeutronWallLoading())
        self.connect("plasmageom.R_min", "q_n_IB.r_in")
        self.connect("plasmageom.surface area", "q_n.SA")
        self.connect("DTfusion.P_n", ["q_n_IB.P_n", "q_n.P_n"])
        self.connect("DTfusion.rate_fus", ["q_n_IB.S"])
        self.connect("q_n_IB.q_n", ["q_n.q_n_IB"])

        # this just needs to be run after the radial build
        self.add_subsystem("blanketgeom", MenardSTBlanketAndShieldGeometry())
        self.connect("plasmageom.a", "blanketgeom.a")
        self.connect("plasmageom.κ", "blanketgeom.κ")
        self.connect("radial_build.props.Ob SOL width",
                "blanketgeom.Ob SOL width")
        self.connect("radial_build.ib.blanket R_max",
                "blanketgeom.Ib blanket R_out")
        self.connect("radial_build.ib.blanket R_min",
                "blanketgeom.Ib blanket R_in")

        self.connect("radial_build.ob.blanket R_in",
                "blanketgeom.Ob blanket R_in")
        self.connect("radial_build.ob.blanket R_out",
                "blanketgeom.Ob blanket R_out")

        self.connect("radial_build.ib.WC shield R_max",
                "blanketgeom.Ib WC shield R_out")
        self.connect("radial_build.ib.WC shield R_min",
                "blanketgeom.Ib WC shield R_in")
        self.connect("radial_build.ib.WC VV shield R_max",
                "blanketgeom.Ib WC VV shield R_out")
        self.connect("radial_build.ib.WC VV shield R_min",
                "blanketgeom.Ib WC VV shield R_in")

        self.add_subsystem("blanket_sh",
                MenardSTBlanketAndShieldMagnetProtection(config=config))
        self.connect("radial_build.props.Ib blanket thickness",
                "blanket_sh.Ib blanket thickness")
        self.connect("radial_build.props.Ib WC shield thickness",
                "blanket_sh.Ib WC shield thickness")
        self.connect("radial_build.props.Ib WC VV shield thickness",
                "blanket_sh.Ib WC VV shield thickness")

        self.add_subsystem("maglife", MenardMagnetLifetime(config=config))
        self.connect("q_n_IB.q_n", "maglife.q_n_IB")
        self.connect("blanket_sh.Shielding factor", "maglife.Shielding factor")



if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True

    prob.model.add_design_var('CS R_max',
                              lower=0.02,
                              upper=0.05,
                              ref=0.3,
                              units='m')
    prob.model.add_design_var('magnets.f_im', lower=0.05, upper=0.95, ref=0.2)
    prob.model.add_design_var(
        'magnets.j_HTS', lower=10, upper=300, ref=150, units="MA/m**2")

    prob.model.add_objective('magnets.B0', scaler=-1)

    # set constraints
    prob.model.add_constraint('magnets.constraint_max_stress', lower=0)
    prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0)
    prob.model.add_constraint('magnets.constraint_wp_current_density',
                              lower=0)
    prob.model.add_constraint('magnets.A_s', lower=0)
    prob.model.add_constraint('magnets.A_m', lower=0)
    prob.model.add_constraint('magnets.A_t', lower=0)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('R0', 3, units='m')
    prob.set_val('aspect ratio', 1.6)
    prob.set_val('magnets.n_coil', 18)
    prob.set_val('magnets.windingpack.j_eff_max', 160, units="MA/m**2")
    prob.set_val('magnets.windingpack.f_HTS', 0.76)
    prob.set_val("magnets.magnetstructure_props.Young's modulus",
                 220, units="GPa")

    # initial values for design variables
    prob.set_val("magnets.f_im", 0.50)
    prob.set_val("magnets.j_HTS", 130, units="MA/m**2")
    prob.set_val("CS R_max", 0.03, units="m")

    # initial inputs for intermediate variables
    prob.set_val('Hbalance.H', 1.7)
    prob.set_val('Ip', 15., units="MA")
    prob.set_val("<n_e>", 1.20, units="n20")
    prob.set_val("radiation.rad.P_loss", 90, units="MW")
    prob.set_val('Bt', 1.8, units='T')

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    # prob.model.nonlinear_solver = om.NonlinearBlockGS()
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.linear_solver = om.DirectSolver()

    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=True,
                                          units=True)
