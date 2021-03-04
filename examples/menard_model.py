# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver.

from openmdao.utils.assert_utils import assert_check_partials

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.radialbuild import MenardSTRadialBuild

from faroes.menardplasmaloop import MenardPlasmaLoop

#from faroes.blanket import MenardSTBlanketAndShieldGeometry
#from faroes.blanket import MenardSTBlanketAndShieldMagnetProtection
#from faroes.blanket import MagnetCryoCoolingPower
#from faroes.blanket import SimpleBlanketPower
#from faroes.blanket import InboardMidplaneNeutronFluxFromRing
#from faroes.blanket import NeutronWallLoading
#from faroes.blanket import MenardMagnetLifetime
#
#from faroes.sol import SOLAndDivertor
#
#from faroes.powerplant import Powerplant

import openmdao.api as om


class TotalRadialBuild(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("radial_build",
                           MenardSTRadialBuild(config=config),
                           promotes_inputs=['CS R_max',
                                            "plasma R_max",
                                            "plasma R_min"])

        self.add_subsystem("magnets",
                           MagnetRadialBuild(config=config),
                           promotes_inputs=["R0"],
                           promotes_outputs=[("B0", "Bt")],
                           )

        self.connect('radial_build.Ob TF R_min', ['magnets.r_iu'])
        self.connect('radial_build.Ib TF R_min', ['magnets.r_is'])
        self.connect('radial_build.Ib TF R_max', ['magnets.r_ot'])

        self.connect('magnets.Ob TF R_out', ['radial_build.Ob TF R_out'])


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("plasmageom",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", ("A", "aspect_ratio")],
                           promotes_outputs=["ε", "κa", "V"])

        self.add_subsystem("totalbuild", TotalRadialBuild(
            config=config), promotes_inputs=["R0"])

        self.connect('plasmageom.R_max', ['totalbuild.plasma R_max'])
        self.connect('plasmageom.R_min', ['totalbuild.plasma R_min'])

        mpl = MenardPlasmaLoop(config=config)
        self.add_subsystem("plasma", mpl,
                           promotes_inputs=["R0", "ε",
                                            "κa", "V",
                                            "aspect_ratio"])
        self.connect("plasmageom.a", "plasma.minor_radius")
        self.connect("plasmageom.L_pol", "plasma.L_pol")
        self.connect("plasmageom.L_pol_simple", "plasma.L_pol_simple")
        self.connect("totalbuild.Bt", "plasma.Bt")

#        # magnet heating model;
#        # total power in blanket
#        self.add_subsystem("magcryo", MagnetCryoCoolingPower(config=config))
#        self.add_subsystem("blanket_P", SimpleBlanketPower(config=config))
#        self.connect("DTfusion.P_n", ["magcryo.P_n", "blanket_P.P_n"])
#
#        # powerplant model
#        self.add_subsystem("pplant", Powerplant(config=config))
#        self.connect("NBIsource.P", ["pplant.P_NBI"])
#        self.connect("NBIsource.eff", ["pplant.η_NBI"])
#        self.connect("DTfusion.P_α", ["pplant.P_α"])
#        self.connect("magcryo.P_c,el", ["pplant.P_cryo"])
#        self.connect("blanket_P.P_th", ["pplant.P_blanket"])
#
#        # SOL and divertor model. Useful for constraints.
#        self.add_subsystem("SOL",
#                           SOLAndDivertor(config=config),
#                           promotes_inputs=["R0", "Bt", "Ip"])
#        self.connect("plasmageom.a", "SOL.a")
#        self.connect("plasmageom.κ", "SOL.κ")
#        self.connect("P_heat.P_heat", "SOL.P_heat")
#
#        # Neutron wall flux models. Useful for constraints.
#        self.add_subsystem("q_n_IB", InboardMidplaneNeutronFluxFromRing(),
#                           promotes_inputs=["R0"])
#        self.add_subsystem("q_n", NeutronWallLoading())
#        self.connect("plasmageom.R_min", "q_n_IB.r_in")
#        self.connect("plasmageom.surface area", "q_n.SA")
#        self.connect("DTfusion.P_n", ["q_n_IB.P_n", "q_n.P_n"])
#        self.connect("DTfusion.rate_fus", ["q_n_IB.S"])
#        self.connect("q_n_IB.q_n", ["q_n.q_n_IB"])
#
#        # this just needs to be run after the radial build
#        self.add_subsystem("blanketgeom", MenardSTBlanketAndShieldGeometry())
#        self.connect("plasmageom.a", "blanketgeom.a")
#        self.connect("plasmageom.κ", "blanketgeom.κ")
#        self.connect("radial_build.props.Ob SOL width",
#                "blanketgeom.Ob SOL width")
#        self.connect("radial_build.ib.blanket R_max",
#                "blanketgeom.Ib blanket R_out")
#        self.connect("radial_build.ib.blanket R_min",
#                "blanketgeom.Ib blanket R_in")
#
#        self.connect("radial_build.ob.blanket R_in",
#                "blanketgeom.Ob blanket R_in")
#        self.connect("radial_build.ob.blanket R_out",
#                "blanketgeom.Ob blanket R_out")
#
#        self.connect("radial_build.ib.WC shield R_max",
#                "blanketgeom.Ib WC shield R_out")
#        self.connect("radial_build.ib.WC shield R_min",
#                "blanketgeom.Ib WC shield R_in")
#        self.connect("radial_build.ib.WC VV shield R_max",
#                "blanketgeom.Ib WC VV shield R_out")
#        self.connect("radial_build.ib.WC VV shield R_min",
#                "blanketgeom.Ib WC VV shield R_in")
#
#        self.add_subsystem("blanket_sh",
#                MenardSTBlanketAndShieldMagnetProtection(config=config))
#        self.connect("radial_build.props.Ib blanket thickness",
#                "blanket_sh.Ib blanket thickness")
#        self.connect("radial_build.props.Ib WC shield thickness",
#                "blanket_sh.Ib WC shield thickness")
#        self.connect("radial_build.props.Ib WC VV shield thickness",
#                "blanket_sh.Ib WC VV shield thickness")
#
#        self.add_subsystem("maglife", MenardMagnetLifetime(config=config))
#        self.connect("q_n_IB.q_n", "maglife.q_n_IB")
#        self.connect("blanket_sh.Shielding factor", "maglife.Shielding factor")


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True

    prob.model.add_design_var('totalbuild.CS R_max',
                              lower=0.02,
                              upper=0.05,
                              ref=0.03,
                              units='m')
    prob.model.add_design_var('aspect_ratio', lower=1.6, upper=5.00, ref=2)
    prob.model.add_design_var(
        'totalbuild.magnets.f_im', lower=0.50, upper=0.95, ref=0.5)
    prob.model.add_design_var(
        'totalbuild.magnets.j_HTS', lower=30, upper=160, ref=150, units="MA/m**2")

    prob.model.add_objective('totalbuild.magnets.B0', scaler=-1)

    # set constraints
    prob.model.add_constraint(
        'totalbuild.magnets.constraint_max_stress', lower=0)
    prob.model.add_constraint(
        'totalbuild.magnets.constraint_B_on_coil', lower=0)
    prob.model.add_constraint(
        'totalbuild.magnets.constraint_wp_current_density', lower=0)
    prob.model.add_constraint('totalbuild.magnets.A_s', lower=0)
    prob.model.add_constraint('totalbuild.magnets.A_m', lower=0)
    prob.model.add_constraint('totalbuild.magnets.A_t', lower=0)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('R0', 3, units='m')
    prob.set_val('aspect_ratio', 2.5)
    prob.set_val('totalbuild.magnets.n_coil', 18)
    prob.set_val('totalbuild.magnets.windingpack.j_eff_max',
                 160, units="MA/m**2")
    prob.set_val('totalbuild.magnets.windingpack.f_HTS', 0.76)
    prob.set_val("totalbuild.magnets.magnetstructure_props.Young's modulus",
                 220, units="GPa")

    # initial values for design variables
    prob.set_val("totalbuild.magnets.f_im", 0.60)
    prob.set_val("totalbuild.magnets.j_HTS", 30, units="MA/m**2")
    prob.set_val("totalbuild.CS R_max", 0.03, units="m")

    build = prob.model.totalbuild
    newton = build.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    build.linear_solver = om.DirectSolver()

    # initial inputs for intermediate variables
    prob.set_val("plasma.Hbalance.H", 1.5)
    prob.set_val("plasma.Ip", 17., units="MA")
    prob.set_val("plasma.<n_e>", 3.20, units="n20")
    # prob.set_val("plasma.confinementtime.PL", 100, units="MW")
    prob.set_val("plasma.radiation.rad.P_loss", 100, units="MW")
    # prob.set_val("plasma.Bt", 6.514, units="T")

    mpl = prob.model.plasma
    newton = mpl.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    mpl.linear_solver = om.DirectSolver()
    mpl.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")
    mpl.linesearch.options["iprint"] = 2
    mpl.linear_solver = om.DirectSolver()

    prob.run_driver()

#    all_inputs = prob.model.list_inputs(values=True,
#                                        print_arrays=True,
#                                        units=True)
#    all_outputs = prob.model.list_outputs(values=True,
#                                          print_arrays=True,
#                                          units=True)
