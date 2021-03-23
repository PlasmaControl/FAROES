# This example includes a Sauter-shape plasma
# and ThreeArcDee coils that fit around it.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.simple_tf_magnet import SimpleMagnetEngineering
from faroes.sauter_plasma import SauterPlasmaGeometryMarginalKappa
from faroes.radialbuild import STRadialBuild

from faroes.util import PolarParallelCurve

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

from faroes.threearcdeecoil import ThreeArcDeeTFSet, ThreeArcDeeTFSetAdaptor

from faroes.sol import SOLAndDivertor

from faroes.powerplant import Powerplant

import openmdao.api as om
import matplotlib.pyplot as plt
import numpy as np


class Geometry(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("radial_build",
                           STRadialBuild(config=config),
                           promotes_inputs=["a"],
                           promotes_outputs=[("A", "aspect_ratio"), "R0",
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

        # Number of angles that will be used for the constraint
        # that prevents overlap between the blanket and TF coils.
        n_θ = 40
        θ = np.linspace(0, 2 * np.pi, n_θ, endpoint=False)
        ivc = om.IndepVarComp()
        ivc.add_output("θ_for_dsq", val=θ)
        self.add_subsystem("ivc", ivc)

        self.add_subsystem("plasma",
                           SauterPlasmaGeometryMarginalKappa(config=config),
                           promotes_inputs=["R0", "a", ("A", "aspect_ratio")],
                           promotes_outputs=["ε", "κ", "κa", "V"])
        self.connect("ivc.θ_for_dsq", "plasma.θ")

        # build exclusion zone
        self.add_subsystem("exclusion_zone",
                           PolarParallelCurve(),
                           promotes_inputs=["R0"])
        self.connect("radial_build.ib.Thermal shield to FW",
                     "exclusion_zone.s")
        self.connect("plasma.R", "exclusion_zone.R")
        self.connect("plasma.Z", "exclusion_zone.Z")
        self.connect("plasma.dR_dθ", "exclusion_zone.dR_dθ")
        self.connect("plasma.dZ_dθ", "exclusion_zone.dZ_dθ")

        # create the 'poloidal shape' of the magnets
        self.add_subsystem("adaptor", ThreeArcDeeTFSetAdaptor())
        self.connect('radial_build.Ob TF R_in', 'adaptor.Ob TF R_in')
        self.connect('plasma.b', 'adaptor.Z_min')
        self.add_subsystem("coils",
                           ThreeArcDeeTFSet(),
                           promotes_inputs=["R0"],
                           promotes_outputs=[])
        self.connect('radial_build.Ib TF R_out',
                     ['adaptor.Ib TF R_out', 'coils.Ib TF R_out'])
        self.connect("adaptor.hhs", "coils.hhs")
        self.connect("adaptor.e_a", "coils.e_a")
        self.connect("adaptor.r_c", "coils.r_c")
        self.connect("exclusion_zone.θ_parall", "coils.θ")

        # compute the space between the exclusion zone and the coils
        self.add_subsystem(
            "margin",
            om.ExecComp(
                "c = a - b",
                a={
                    "units": "m**2",
                    "shape_by_conn": True
                },
                b={
                    "units": "m**2",
                    "shape_by_conn": True,
                    "copy_shape": "a"
                },
                c={
                    "units": "m**2",
                    "copy_shape": "a"
                },
                has_diag_partials=True,
            ))

        self.connect("coils.d_sq", "margin.a")
        self.connect("exclusion_zone.d_sq", "margin.b")

        # combine the vector of 'margin' into a single constraint,
        # using the Kreisselmeier-Steinhauser Function.
        # I think this is basically a 'softmax' function.
        # Note that this is the only time there is a constraint added other
        # than in the main script below.
        self.add_subsystem(
            'ks',
            om.KSComp(width=n_θ,
                      units="m**2",
                      ref=1,
                      lower_flag=True,
                      rho=10,
                      upper=0,
                      add_constraint=True))
        self.connect("margin.c", "ks.g")

        # Additional geometry
        # computes how well the magnets are protected from neutron damage
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

        # Zero-D plasma model with NBI
        mpl = MenardPlasmaLoop(config=config)
        self.add_subsystem(
            "plasma",
            mpl,
            promotes_inputs=["R0", "ε", "κa", ("minor_radius", "a")],
        )
        self.connect("geometry.V", "plasma.V")
        self.connect("geometry.aspect_ratio", ["plasma.aspect_ratio"])
        self.connect("geometry.plasma.L_pol",
                     ["plasma.L_pol", "plasma.L_pol_simple"])

        self.connect("magnets.B0", ["plasma.Bt"])

        # magnet heating model;
        self.add_subsystem("magcryo", MagnetCryoCoolingPower(config=config))
        self.connect("geometry.blanket_sh.Eff Sh+Bl n thickness",
                     "magcryo.Δr_sh")

        # total thermal power generated the in blanket
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

        self.connect("geometry.plasma.S", "q_n.SA")
        self.connect("plasma.DTfusion.P_n", ["q_n_IB.P_n", "q_n.P_n"])
        self.connect("plasma.DTfusion.rate_fus", ["q_n_IB.S"])
        self.connect("q_n_IB.q_n", ["q_n.q_n_IB"])

        # compute magnet lifetime w.r.t. neutron damage
        self.add_subsystem("maglife", MenardMagnetLifetime(config=config))
        self.connect("q_n_IB.q_n", "maglife.q_n_IB")
        self.connect("geometry.blanket_sh.Shielding factor",
                     "maglife.Shielding factor")

        # define a custom objective. Here we add in the enclosed volume of the
        # TF coils, so that there is an incentive to shrink the coils.
        # This objective is NOT dimensionally coherent.
        self.add_subsystem(
            "custom_obj",
            om.ExecComp("obj = Paux + 0.3 * Venc",
                        Paux={'units': 'MW'},
                        Venc={'units': 'm**3'},
                        obj={'ref': 1000}))
        self.connect('pplant.aux.P_aux,h', 'custom_obj.Paux')
        self.connect('geometry.coils.V_enc', 'custom_obj.Venc')


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    machine = Machine(config=uc)
    prob.model = machine

    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output("a", units="m")
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

    prob.driver = om.pyOptSparseDriver()

    prob.driver.options['optimizer'] = 'IPOPT'
    prob.driver.options['user_terminate_signal'] = True

    prob.model.add_design_var('geometry.radial_build.CS ΔR',
                              lower=0.10,
                              upper=0.5,
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
    prob.model.add_design_var('a', lower=0.9, upper=3, units='m')
    prob.model.add_design_var('geometry.radial_build.ob.gap thickness',
                              lower=0.00,
                              upper=5.0,
                              ref=0.3,
                              units='m')
    prob.model.add_design_var('geometry.adaptor.f_c', lower=0.01, upper=0.99)
    prob.model.add_design_var('geometry.adaptor.Z_1',
                              lower=0.0,
                              upper=10,
                              units='m')
    prob.model.add_design_var('magnets.j_HTS',
                              lower=25,
                              upper=150,
                              ref=100,
                              units="MA/m**2")
    prob.model.add_design_var('plasma.NBIsource.P',
                              lower=50,
                              upper=350,
                              ref=100,
                              units="MW")

    # prob.model.add_objective('pplant.overall.P_net', scaler=-1)
    # prob.model.add_objective('pplant.overall.f_recirc', scaler=1)
    # prob.model.add_objective('pplant.aux.P_aux,h', scaler=1)
    # prob.model.add_objective('magnets.B0', scaler=-1)
    # prob.model.add_objective('geometry.radial_build.cryostat R_out',
    #    scaler=1)
    prob.model.add_objective('custom_obj.obj', scaler=1)

    # set constraints
    prob.model.add_constraint('magnets.constraint_max_stress',
                              lower=0,
                              upper=0.05)
    prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0, upper=1)
    prob.model.add_constraint('magnets.constraint_wp_current_density', lower=0)
    prob.model.add_constraint('R0', lower=3.0, upper=9.0, units='m')
    prob.model.add_constraint('geometry.aspect_ratio', lower=1.8, upper=3)
    prob.model.add_constraint('plasma.Hbalance.H', lower=0.99, upper=1.01)
    prob.model.add_constraint('geometry.adaptor.r_c', lower=0.2)
    prob.model.add_constraint('pplant.overall.P_net',
                              lower=599,
                              upper=601,
                              units='MW')

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('geometry.radial_build.ib_tf.n_coil', 18)

    # initial values for design variables
    prob.set_val("geometry.radial_build.CS ΔR", 0.45, units="m")
    prob.set_val("geometry.radial_build.ib_tf.Δr_s", 0.25, units='m')
    prob.set_val("geometry.radial_build.ib_tf.Δr_m", 0.25, units='m')
    prob.set_val('a', 1.7, units="m")
    prob.set_val("magnets.j_HTS", 70, units="MA/m**2")
    prob.set_val("plasma.NBIsource.P", 280, units="MW")
    prob.set_val("geometry.adaptor.f_c", 0.9)
    prob.set_val("geometry.adaptor.Z_1", -2.0)

    # parameters
    prob.set_val("geometry.plasma.δ", 0.3)

    # initial inputs for intermediate variables
    prob.set_val("plasma.Hbalance.H", 1.00)
    prob.set_val("plasma.Ip", 28., units="MA")
    prob.set_val("plasma.<n_e>", 2.3, units="n20")
    prob.set_val("plasma.radiation.rad.P_loss", 683, units="MW")

    # start solvers
    build = prob.model.geometry
    newton = build.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    build.linear_solver = om.DirectSolver()

    mpl = prob.model.plasma
    newton = mpl.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 8
    mpl.linear_solver = om.DirectSolver()
    newton.linesearch = om.ArmijoGoldsteinLS(retry_on_analysis_error=True,
                                             rho=0.5,
                                             c=0.1,
                                             method="Armijo",
                                             bound_enforcement="vector")
    newton.linesearch.options["maxiter"] = 40

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

    fig, ax = plt.subplots()
    machine.geometry.plasma.geom.plot(ax)
    machine.geometry.exclusion_zone.pts.plot(ax)
    machine.geometry.coils.plot(ax)
    ax.set_xlim([-1, 8])
    ax.axis('equal')
    plt.show()
