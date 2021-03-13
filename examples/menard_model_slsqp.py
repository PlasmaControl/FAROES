# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.radialbuild import MenardSTRadialBuild

from faroes.menardplasmaloop import MenardPlasmaLoop

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
                           MenardSTRadialBuild(config=config),
                           promotes_inputs=["a", "CS R_max", "Ib TF R_max"],
                           promotes_outputs=[("A", "aspect_ratio"), "R0", "Ib TF R_min",
                               "plasma R_max", "plasma R_min"])

        self.add_subsystem("magnets",
                           MagnetRadialBuild(config=config),
                           promotes_inputs=["R0", ("r_is", "Ib TF R_min")],
                           promotes_outputs=[("Ib TF R_out", "Ib TF R_max"),
                           ("B0", "Bt")])

        self.connect('radial_build.Ob TF R_min', ['magnets.r_iu'])

        self.connect('magnets.Ob TF R_out', ['radial_build.Ob TF R_out'])

        self.add_subsystem("plasma",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", "a", ("A", "aspect_ratio")],
                           promotes_outputs=["ε", "κ", "κa", "V"])

        self.add_subsystem(
            "blanket_sh",
            MenardSTBlanketAndShieldMagnetProtection(config=config))
        self.connect("radial_build.ib_blanket.blanket_thickness",
                     "blanket_sh.Ib blanket thickness")
        self.connect("radial_build.ib_shield.shield_thickness",
                     "blanket_sh.Ib WC shield thickness")
        self.connect("radial_build.props.Ib WC VV shield thickness",
                     "blanket_sh.Ib WC VV shield thickness")

        # this just needs to be run after the radial build
        self.add_subsystem("blanketgeom", MenardSTBlanketAndShieldGeometry(),
                promotes_inputs=["a", "κ"])
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



class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']


        self.add_subsystem("geometry",
                           Geometry(config=config),
                           promotes_inputs=["a"],
                           promotes_outputs=["R0", "κ", "κa", "V", "ε"])

        mpl = MenardPlasmaLoop(config=config)
        self.add_subsystem(
            "plasma",
            mpl,
            promotes_inputs=["R0", "ε", "κa", "V",
            ("minor_radius", "a")],
        )
        self.connect("geometry.aspect_ratio", ["plasma.aspect_ratio"])
        self.connect("geometry.plasma.L_pol", "plasma.L_pol")
        self.connect("geometry.plasma.L_pol_simple", "plasma.L_pol_simple")

        self.connect("geometry.Bt", ["plasma.Bt", "SOL.Bt"])

        self.connect("geometry.blanket_sh.Eff Sh+Bl n thickness", "magcryo.Δr_sh")

        # magnet heating model;
        # total power in blanket
        self.add_subsystem("magcryo", MagnetCryoCoolingPower(config=config))
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

        # Neutron wall flux models. Useful for constraints.
        self.add_subsystem("q_n_IB",
                           InboardMidplaneNeutronFluxFromRing(),
                           promotes_inputs=["R0"])
        self.add_subsystem("q_n", NeutronWallLoading())
        self.connect("geometry.plasma R_min", "q_n_IB.r_in")

        self.connect("geometry.plasma.surface area", "q_n.SA")
        self.connect("plasma.DTfusion.P_n", ["q_n_IB.P_n", "q_n.P_n"])
        self.connect("plasma.DTfusion.rate_fus", ["q_n_IB.S"])
        self.connect("q_n_IB.q_n", ["q_n.q_n_IB"])

        self.add_subsystem("maglife", MenardMagnetLifetime(config=config))
        self.connect("q_n_IB.q_n", "maglife.q_n_IB")
        self.connect("geometry.blanket_sh.Shielding factor", "maglife.Shielding factor")


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output("a", 1.5, units="m")
    prob.model.add_subsystem("P", ivc, promotes_outputs=["*"])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['disp'] = True

    prob.driver.options['optimizer'] = 'SLSQP'

    prob.model.add_design_var('geometry.CS R_max',
                              lower=0.10,
                              upper=1.000,
                              ref=0.50,
                              units='m')
    prob.model.add_design_var('geometry.magnets.Δr_s', lower=0.05, upper=1.5, ref=0.3)
    prob.model.add_design_var('geometry.magnets.Δr_m', lower=0.05, upper=1.5, ref=0.3)
    prob.model.add_design_var('a', lower=0.3, upper=3, units='m')
    prob.model.add_design_var('geometry.magnets.j_HTS',
                              lower=10,
                              upper=250,
                              ref=100,
                              units="MA/m**2")

    # prob.model.add_objective('geometry.magnets.B0', scaler=-1)
    # prob.model.add_constraint('pplant.overall.P_net', lower=400)
    prob.model.add_objective('pplant.overall.P_net', scaler=-1)

    # set constraints
    prob.model.add_constraint('geometry.magnets.constraint_max_stress',
                              lower=0)
    prob.model.add_constraint('geometry.magnets.constraint_B_on_coil',
                              lower=0)
    prob.model.add_constraint(
        'geometry.magnets.constraint_wp_current_density', lower=0)
    prob.model.add_constraint('R0', equals=3.0)
    prob.model.add_constraint('geometry.aspect_ratio', lower=1.6, upper=5)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('geometry.magnets.n_coil', 18)

    # initial values for design variables
    prob.set_val("geometry.CS R_max", 0.15, units="m")
    prob.set_val("geometry.magnets.Δr_s", 0.1, units='m')
    prob.set_val("geometry.magnets.Δr_m", 0.1, units='m')
    prob.set_val('a', 1.5, units="m")
    prob.set_val("geometry.magnets.j_HTS", 20, units="MA/m**2")

    build = prob.model.geometry
    newton = build.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    build.linear_solver = om.DirectSolver()
    newton.linesearch = om.ArmijoGoldsteinLS(retry_on_analysis_error=True,
            rho=0.5, c=0.10, method="Armijo", bound_enforcement="vector")
    newton.linesearch.options["maxiter"] = 10
    newton.linesearch.options["iprint"] = 0

    # initial inputs for intermediate variables
    prob.set_val("plasma.Hbalance.H", 1.30)
    prob.set_val("plasma.Ip", 10., units="MA")
    prob.set_val("plasma.<n_e>", 1.0 , units="n20")
    prob.set_val("plasma.radiation.rad.P_loss", 100, units="MW")

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
            rho=0.5, c=0.03, method="Armijo", bound_enforcement="vector")
    newton.linesearch.options["maxiter"] = 40
    newton.linesearch.options["iprint"] = 2
    newton.linesearch.options["print_bound_enforce"] = False

    pplant = prob.model.pplant
    newton = pplant.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 10
    pplant.linear_solver = om.DirectSolver()
    newton.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")
    newton.linesearch.options["iprint"] = 0

    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=True,
                                          units=True)
