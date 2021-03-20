# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.simple_tf_magnet import SimpleMagnetEngineering
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.radialbuild import STRadialBuild

from faroes.menardplasmaloop import MenardPlasmaLoop

from faroes.blanket import MenardInboardBlanketFit
from faroes.blanket import MenardInboardShieldFit

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


class Geometry(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("radial_build",
                           STRadialBuild(config=config),
                           promotes_inputs=["a"],
                           promotes_outputs=[("A", "aspect_ratio"), "R0",
                                             "Ib TF R_in", "plasma R_out",
                                             "plasma R_in"])

        self.add_subsystem("ib_shield",
                           MenardInboardShieldFit(config=config),
                           promotes_inputs=[("A", "aspect_ratio")])
        self.add_subsystem("ib_blanket",
                           MenardInboardBlanketFit(config=config),
                           promotes_inputs=[("A", "aspect_ratio")])

        self.connect('ib_shield.shield_thickness',
                     ['radial_build.ib.WC shield thickness'])
        self.connect('ib_blanket.blanket_thickness',
                     ['radial_build.ib.blanket thickness'])

        self.add_subsystem("plasma",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", "a", ("A", "aspect_ratio")],
                           promotes_outputs=["ε", "κ", "κa", "V"])

        self.add_subsystem(
            "blanket_sh",
            MenardSTBlanketAndShieldMagnetProtection(config=config))
        self.connect("ib_blanket.blanket_thickness",
                     "blanket_sh.Ib blanket thickness")
        self.connect("ib_shield.shield_thickness",
                     "blanket_sh.Ib WC shield thickness")
        self.connect("radial_build.props.Ib WC VV shield thickness",
                     "blanket_sh.Ib WC VV shield thickness")

        # this just needs to be run after the radial build
        self.add_subsystem("blanketgeom",
                           MenardSTBlanketAndShieldGeometry(),
                           promotes_inputs=["a", "κ"])
        self.connect("radial_build.props.Ob SOL width",
                     "blanketgeom.Ob SOL width")
        self.connect("radial_build.ib.blanket R_out",
                     "blanketgeom.Ib blanket R_out")
        self.connect("radial_build.ib.blanket R_in",
                     "blanketgeom.Ib blanket R_in")

        self.connect("radial_build.ob.blanket R_in",
                     "blanketgeom.Ob blanket R_in")
        self.connect("radial_build.ob.blanket R_out",
                     "blanketgeom.Ob blanket R_out")

        self.connect("radial_build.ib.WC shield R_out",
                     "blanketgeom.Ib WC shield R_out")
        self.connect("radial_build.ib.WC shield R_in",
                     "blanketgeom.Ib WC shield R_in")
        self.connect("radial_build.ib.WC VV shield R_out",
                     "blanketgeom.Ib WC VV shield R_out")
        self.connect("radial_build.ib.WC VV shield R_in",
                     "blanketgeom.Ib WC VV shield R_in")


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("geometry",
                           Geometry(config=config),
                           promotes_inputs=["a"],
                           promotes_outputs=["R0", "κ", "κa", "ε"])

        self.add_subsystem("magnets",
                           SimpleMagnetEngineering(config=config),
                           promotes_inputs=["R0"])

        self.connect('geometry.radial_build.ib_tf.r1', ['magnets.r1'])
        self.connect('geometry.radial_build.ob_tf.r2', ['magnets.r2'])
        self.connect('geometry.radial_build.ib_tf.A_s', ['magnets.A_s'])
        self.connect('geometry.radial_build.ib_tf.A_m', ['magnets.A_m'])
        self.connect('geometry.radial_build.ib_tf.A_t', ['magnets.A_t'])
        self.connect('geometry.radial_build.ib_tf.r_om',
                     ['magnets.Ib winding pack R_out'])

        mpl = MenardPlasmaLoop(config=config)
        self.add_subsystem(
            "plasma",
            mpl,
            promotes_inputs=["R0", "ε", "κa", ("minor_radius", "a")],
        )
        self.connect("geometry.V", "plasma.V")
        self.connect("geometry.aspect_ratio", ["plasma.aspect_ratio"])
        self.connect("geometry.plasma.L_pol", "plasma.L_pol")
        self.connect("geometry.plasma.L_pol_simple", "plasma.L_pol_simple")

        self.connect("magnets.B0", ["plasma.Bt"])

        # magnet heating model;
        # total power in blanket
        self.add_subsystem("magcryo", MagnetCryoCoolingPower(config=config))
        self.connect("geometry.blanket_sh.Eff Sh+Bl n thickness",
                     "magcryo.Δr_sh")
        self.add_subsystem("blanket_P", SimpleBlanketPower(config=config))
        self.connect("plasma.DTfusion.P_n", ["magcryo.P_n", "blanket_P.P_n"])

        # powerplant model
        self.add_subsystem("pplant", Powerplant(config=config))
        self.connect("plasma.NBIsource.P", ["pplant.P_NBI"])
        self.connect("plasma.NBIsource.eff", ["pplant.η_NBI"])
        self.connect("plasma.DTfusion.P_α", ["pplant.P_α"])
        self.connect("magcryo.P_c,el", ["pplant.P_cryo"])
        self.connect("blanket_P.P_th", ["pplant.P_blanket"])

        # SOL and divertor model. Useful for constraints.
        self.add_subsystem("SOL",
                           SOLAndDivertor(config=config),
                           promotes_inputs=["R0", "κ", "a"])
        self.connect("plasma.P_heat.P_heat", "SOL.P_heat")
        self.connect("plasma.Ip", "SOL.Ip")
        self.connect("plasma.current.q_star", "SOL.q_star")
        self.connect("magnets.B0", ["SOL.Bt"])

        # Neutron wall flux models. Useful for constraints.
        self.add_subsystem("q_n_IB",
                           InboardMidplaneNeutronFluxFromRing(),
                           promotes_inputs=["R0"])
        self.add_subsystem("q_n", NeutronWallLoading())
        self.connect("geometry.plasma R_in", "q_n_IB.r_in")

        self.connect("geometry.plasma.surface area", "q_n.SA")
        self.connect("plasma.DTfusion.P_n", ["q_n_IB.P_n", "q_n.P_n"])
        self.connect("plasma.DTfusion.rate_fus", ["q_n_IB.S"])
        self.connect("q_n_IB.q_n", ["q_n.q_n_IB"])

        self.add_subsystem("maglife", MenardMagnetLifetime(config=config))
        self.connect("q_n_IB.q_n", "maglife.q_n_IB")
        self.connect("geometry.blanket_sh.Shielding factor",
                     "maglife.Shielding factor")


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output("a", units="m")
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['disp'] = True

    prob.driver.options['optimizer'] = 'SLSQP'

    prob.model.add_design_var('geometry.radial_build.CS ΔR',
                              lower=0.10,
                              upper=1.0,
                              ref=0.50,
                              units='m')
    prob.model.add_design_var('geometry.radial_build.ib_tf.Δr_s',
                              lower=0.05,
                              upper=1.0,
                              ref=0.3,
                              units='m')
    prob.model.add_design_var('geometry.radial_build.ib_tf.Δr_m',
                              lower=0.05,
                              upper=1.0,
                              ref=0.3,
                              units='m')
    prob.model.add_design_var('a', lower=0.9, upper=2, units='m')
    prob.model.add_design_var('geometry.radial_build.ob.gap thickness',
                              lower=0.00,
                              upper=1.0,
                              ref=0.3,
                              units='m')
    prob.model.add_design_var('magnets.j_HTS',
                              lower=10,
                              upper=200,
                              ref=100,
                              units="MA/m**2")

    prob.model.add_objective('pplant.overall.P_net', scaler=-1)
    # prob.model.add_objective('magnets.B0', scaler=-1)
    # prob.model.add_objective('geometry.radial_build.cryostat R_out',
    #                           scaler=-1)

    # set constraints
    prob.model.add_constraint('magnets.constraint_max_stress', lower=0)
    prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0)
    prob.model.add_constraint('magnets.constraint_wp_current_density', lower=0)
    prob.model.add_constraint('R0', equals=3.0)
    prob.model.add_constraint('geometry.aspect_ratio', lower=1.6, upper=5)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('geometry.radial_build.ib_tf.n_coil', 18)

    # initial values for design variables
    prob.set_val("geometry.radial_build.CS ΔR", 0.15, units="m")
    prob.set_val("geometry.radial_build.ib_tf.Δr_s", 0.1, units='m')
    prob.set_val("geometry.radial_build.ib_tf.Δr_m", 0.1, units='m')
    prob.set_val('a', 1.5, units="m")
    prob.set_val("magnets.j_HTS", 20, units="MA/m**2")

    # initial inputs for intermediate variables
    prob.set_val("plasma.Hbalance.H", 1.70)
    prob.set_val("plasma.Ip", 15., units="MA")
    prob.set_val("plasma.<n_e>", 1.5, units="n20")
    prob.set_val("plasma.radiation.rad.P_loss", 140, units="MW")

    # set up solvers
    build = prob.model.geometry
    newton = build.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    build.linear_solver = om.DirectSolver()
    newton.linesearch = om.ArmijoGoldsteinLS(retry_on_analysis_error=True,
                                             rho=0.5,
                                             c=0.10,
                                             method="Armijo",
                                             bound_enforcement="vector")
    newton.linesearch.options["maxiter"] = 10
    newton.linesearch.options["iprint"] = 0

    mpl = prob.model.plasma
    newton = mpl.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 100
    newton.options['rtol'] = 1e-4
    newton.options['atol'] = 1e-4
    newton.options['stall_tol'] = 1e-5
    newton.options['stall_limit'] = 10
    mpl.linear_solver = om.DirectSolver()
    newton.linesearch = om.ArmijoGoldsteinLS(retry_on_analysis_error=True,
                                             rho=0.5,
                                             c=0.03,
                                             method="Armijo",
                                             bound_enforcement="vector")
    newton.linesearch.options["maxiter"] = 40
    newton.linesearch.options["iprint"] = 2
    newton.linesearch.options["print_bound_enforce"] = False

    pplant = prob.model.pplant
    newton = pplant.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 10
    pplant.linear_solver = om.DirectSolver()

    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=True,
                                          units=True)
