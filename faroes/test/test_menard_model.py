# This tests a decently-large chunk of the Menard spreadsheet model

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

from openmdao.utils.assert_utils import assert_near_equal
import unittest


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
                           promotes_outputs=["??", "??", "??a", "V"])

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
                           promotes_inputs=["a", "??"])
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
                           promotes_outputs=["R0", "??", "??a", "??"])

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
            promotes_inputs=["R0", "??", "??a", ("minor_radius", "a")],
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
                     "magcryo.??r_sh")
        self.add_subsystem("blanket_P", SimpleBlanketPower(config=config))
        self.connect("plasma.DTfusion.P_n", ["magcryo.P_n", "blanket_P.P_n"])

        # powerplant model
        self.add_subsystem("pplant", Powerplant(config=config))
        self.connect("plasma.NBIsource.P", ["pplant.P_NBI"])
        self.connect("plasma.NBIsource.eff", ["pplant.??_NBI"])
        self.connect("plasma.DTfusion.P_??", ["pplant.P_??"])
        self.connect("magcryo.P_c,el", ["pplant.P_cryo"])
        self.connect("blanket_P.P_th", ["pplant.P_blanket"])

        # SOL and divertor model. Useful for constraints.
        self.add_subsystem("SOL",
                           SOLAndDivertor(config=config),
                           promotes_inputs=["R0", "??", "a"])
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


class TestMenardModel(unittest.TestCase):
    def setUp(self):
        prob = om.Problem()

        uc = UserConfigurator()
        prob.model = Machine(config=uc)

        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output("a", units="m")
        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        model.add_design_var('geometry.radial_build.CS ??R',
                             lower=0.10,
                             upper=1.0,
                             ref=0.50,
                             units='m')
        model.add_design_var('geometry.radial_build.ib_tf.??r_s',
                             lower=0.05,
                             upper=1.0,
                             ref=0.3,
                             units='m')
        model.add_design_var('geometry.radial_build.ib_tf.??r_m',
                             lower=0.05,
                             upper=1.0,
                             ref=0.3,
                             units='m')
        model.add_design_var('a', lower=0.9, upper=3, units='m')
        model.add_design_var('geometry.radial_build.ob.gap thickness',
                             lower=0.00,
                             upper=5.0,
                             ref=0.3,
                             units='m')
        model.add_design_var('magnets.j_HTS',
                             lower=10,
                             upper=250,
                             ref=100,
                             units="MA/m**2")
        model.add_objective('pplant.overall.P_net', scaler=-1.0, units="GW")

        # set constraints
        model.add_constraint('magnets.constraint_max_stress', lower=0, ref=0.1)
        model.add_constraint('magnets.constraint_B_on_coil', lower=0)
        model.add_constraint('magnets.constraint_wp_current_density',
                             lower=0,
                             ref=20)
        model.add_constraint('R0', equals=3.0)
        model.add_constraint('geometry.aspect_ratio',
                             lower=1.6,
                             upper=5,
                             ref=3)

        prob.setup()
        prob.check_config(checks=['unconnected_inputs'])

        prob.set_val('geometry.radial_build.ib_tf.n_coil', 18)

        # initial values for design variables
        prob.set_val("geometry.radial_build.CS ??R", 0.20, units="m")
        prob.set_val("geometry.radial_build.ib_tf.??r_s", 0.28, units='m')
        prob.set_val("geometry.radial_build.ib_tf.??r_m", 0.14, units='m')
        prob.set_val('a', 1.53, units="m")
        prob.set_val("magnets.j_HTS", 155, units="MA/m**2")

        build = model.geometry
        newton = build.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True)
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 20
        build.linear_solver = om.DirectSolver()

        # initial inputs for intermediate variables
        prob.set_val("plasma.Hbalance.H", 1.70)
        prob.set_val("plasma.Ip", 15., units="MA")
        prob.set_val("plasma.<n_e>", 1.625, units="n20")
        prob.set_val("plasma.radiation.rad.P_loss", 167, units="MW")

        mpl = model.plasma
        newton = mpl.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 20
        mpl.linear_solver = om.DirectSolver()
        newton.linesearch = om.ArmijoGoldsteinLS(retry_on_analysis_error=True,
                                                 rho=0.5,
                                                 c=0.1,
                                                 method="Armijo",
                                                 bound_enforcement="vector")
        newton.linesearch.options["maxiter"] = 40

        pplant = prob.model.pplant
        newton = pplant.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True)
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        pplant.linear_solver = om.DirectSolver()
        self.prob = prob

    def test_values(self):
        prob = self.prob
        prob.run_driver()

        f = prob.get_val("plasma.Hbalance.H")
        expected = 1.69793
        assert_near_equal(f, expected, tolerance=1e-3)
        f = prob.get_val("magnets.B0", units="T")
        expected = 3.885
        assert_near_equal(f, expected, tolerance=1e-3)


if __name__ == "__main__":
    unittest.main()
