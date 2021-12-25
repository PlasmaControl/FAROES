from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.elliptical_plasma import MenardPlasmaGeometry

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


class MenardPlasmaLoop(om.Group):
    r"""
    Inputs
    ------
    R0 : float
        m, Plasma major radius.
        There is no Shafranov shift in this model
    aspect_ratio : float
        Aspect ratio
    minor_radius : float
        m, Minor radius
    ε : float
        Inverse aspect ratio
    δ : float
        Triangularity
    κa : float
        Effective kappa
    V : float
        m**3, Plasma volume

    L_pol : float
        Poloidal circumfirence (exact ellipse formula)
    L_pol_simple : float
        Simplified poloidal circumfirence (simple ellipse perimeter)
    Bt : float
        T, Toroidal field

    Notes
    -----
    Required initialization variables:

    Ip : float
        MA, Plasma current. Typically ~10
    Hbalance.H : float
        The H-factor. Typically between 1 to 1.5.
    <n_e> : float
        n20, averaged electron density. Typically 1 to 1.5.
    radiation.rad.P_loss : float
        MW, Power lost via particle diffusion into the SOL.
    """
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        acc = self.options["config"].accessor(["fits", "τe"])
        scaling = acc(["default"])

        self.add_subsystem("confinementtime",
                           ConfinementTime(config=config, scaling=scaling),
                           promotes_inputs=[("R", "R0"), "ε", "κa", "Bt",
                                            ("n19", "<n_e>"), "Ip"])

        self.add_subsystem("ZeroDPlasma",
                           ZeroDPlasma(config=config),
                           promotes_inputs=["<n_e>", "V"],
                           promotes_outputs=["<T_e>"])

        self.connect("confinementtime.τe", ["ZeroDPlasma.τ_th"])
        self.connect("ZeroDPlasma.A", "confinementtime.M")

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
            om.ExecComp(
                ["P_heat = P_alpha + P_NBI + P_RF", "P_aux = P_NBI + P_RF"],
                P_heat={
                    'units': 'MW',
                    'desc': "Total heating power"
                },
                P_aux={
                    'units': 'MW',
                    'desc': "Aux. heating power"
                },
                P_alpha={
                    'units': 'MW',
                    'desc': "Alpha heating power"
                },
                P_RF={
                    'units': 'MW',
                    'val': 0,
                    'desc': "RF aux. heating power"
                },
                P_NBI={
                    'units': 'MW',
                    'val': 0,
                    'desc': "neutral beam aux. heating power"
                }))
        self.connect("DTfusion.P_α", "P_heat.P_alpha")
        self.connect("NBIsource.P", "P_heat.P_NBI")

        self.add_subsystem("radiation", SimpleRadiation(config=config))
        self.connect("P_heat.P_heat", "radiation.P_heat")

        # back-connection
        self.connect("radiation.P_loss",
                     ["ZeroDPlasma.P_loss", "confinementtime.PL"])

        self.add_subsystem("specP",
                           SpecifiedPressure(config=config),
                           promotes_inputs=[
                               "Bt", "Ip", ("a", "minor_radius"),
                               ("L_pol", "L_pol"), ("A", "aspect_ratio")
                           ])

        # self.connect("plasmageom.a", ["specP.a"])
        # self.connect("plasmageom.L_pol", ["specP.L_pol"])

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
                           promotes_inputs=["ε", "δ", "Ip"])
        self.connect("ZeroDPlasma.thermal pressure fraction",
                     "bootstrap.thermal pressure fraction")
        self.connect("specP.βp", "bootstrap.βp")

        self.add_subsystem('current',
                           CurrentAndSafetyFactor(config=config),
                           promotes_inputs=[
                               "R0", "Bt", ("a", "minor_radius"),
                               ("L_pol", "L_pol_simple")
                           ],
                           promotes_outputs=["Ip", ("n_bar", "<n_e>")])

        self.connect("NBIcurr.I_NBI", "current.I_NBI")
        self.connect("bootstrap.I_BS", "current.I_BS")
        # back-connections
        self.connect("current.q_star", "bootstrap.q_star")
        self.connect("current.q_min", "bootstrap.q_min")

        self.add_subsystem(
            "Q_phys",
            om.ExecComp("Q = P_fus/P_heat",
                        Q={'desc': "Plasma gain"},
                        P_fus={
                            'units': 'MW',
                            'desc': "Primary fusion power"
                        },
                        P_heat={
                            'units': 'MW',
                            'desc': "Auxilliary heating power"
                        }))
        self.connect("P_heat.P_aux", "Q_phys.P_heat")
        self.connect("DTfusion.P_fus", "Q_phys.P_fus")


if __name__ == "__main__":

    class Machine(om.Group):
        def initialize(self):
            self.options.declare('config')

        def setup(self):
            config = self.options['config']

            self.add_subsystem("plasmageom",
                               MenardPlasmaGeometry(config=config),
                               promotes_inputs=["R0", "A", "a"],
                               promotes_outputs=["ε", "κa", "V"])

            mpl = MenardPlasmaLoop(config=config)
            self.add_subsystem("plasma",
                               mpl,
                               promotes_inputs=[("minor_radius", "a"), "R0",
                                                "Bt", "ε", "κa", "V",
                                                ("aspect_ratio", "A")])
            self.connect("plasmageom.L_pol", "plasma.L_pol")
            self.connect("plasmageom.L_pol_simple", "plasma.L_pol_simple")

    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.setup()

    prob.set_val('R0', 3, units='m')
    prob.set_val('a', 1.875, units='m')
    prob.set_val('A', 1.6)
    prob.set_val('Bt', 2.094, units='T')

    # initial inputs
    prob.set_val('plasma.Hbalance.H', 1.77)
    prob.set_val('plasma.Ip', 15.28, units="MA")
    prob.set_val("plasma.<n_e>", 1.107, units="n20")
    prob.set_val("plasma.radiation.rad.P_loss", 92.2, units="MW")

    mpl = prob.model.plasma
    newton = mpl.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    mpl.linear_solver = om.DirectSolver()

    # initial values

    prob.run_driver()

    all_inputs = prob.model.list_inputs(val=True,
                                        print_arrays=True,
                                        units=True,
                                        desc=True)
    all_outputs = prob.model.list_outputs(val=True,
                                          print_arrays=True,
                                          units=True,
                                          desc=True)
