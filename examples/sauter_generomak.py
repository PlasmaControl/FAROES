# This example includes a Sauter-shape plasma
# and the Generomak costing model.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.radialbuild import STRadialBuild
from faroes.sauter_plasma import SauterPlasmaGeometryMarginalKappa
from faroes.simple_tf_magnet import SimpleMagnetEngineering

from faroes.cryostat import SimpleCryostat

from faroes.util import PolarParallelCurve

from faroes.rfheating import SimpleRFHeating
from faroes.menardplasmaloop import MenardPlasmaLoop

from faroes.blanket import MenardInboardBlanketFit
from faroes.blanket import MenardInboardShieldFit
from faroes.blanket import OutboardBlanketFit

from faroes.blanket import MenardSTBlanketAndShieldGeometry
from faroes.blanket import MenardSTBlanketAndShieldMagnetProtection
from faroes.blanket import MagnetCryoCoolingPower
from faroes.blanket import SimpleBlanketPower
from faroes.blanket import InboardMidplaneNeutronFluxFromRing
from faroes.blanket import NeutronWallLoading
from faroes.blanket import MenardMagnetLifetime

from faroes.threearcdeecoil import ThreeArcDeeTFSet, ThreeArcDeeTFSetAdaptor
from faroes.tf_magnet_set import TFMagnetSet

from faroes.sol import SOLAndDivertor

from faroes.ripple import SimpleRipple

from faroes.powerplant import Powerplant

from faroes.generomakcosting import GeneromakCosting

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
        self.add_subsystem("ob_blanket",
                           OutboardBlanketFit(config=config),
                           promotes_inputs=[("A", "aspect_ratio")])

        self.connect('ib_shield.shield_thickness',
                     ['radial_build.ib.WC shield thickness'])
        self.connect('ib_blanket.blanket_thickness',
                     ['radial_build.ib.blanket thickness'])
        self.connect('ob_blanket.blanket_thickness',
                     ['radial_build.ob.blanket thickness'])

        self.add_subsystem("ripple", SimpleRipple())
        self.connect("radial_build.ib_tf.r1", "ripple.r1")
        self.connect("radial_build.ob_tf.r2", "ripple.r2")
        self.connect("radial_build.plasma R_out", "ripple.R")

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

        # build fw shape
        self.add_subsystem("fw_shape",
                           PolarParallelCurve(use_Rmin=True, torus_V=True),
                           promotes_inputs=["R0"])
        self.connect("radial_build.ob_plasma_to_x.ob_pl_to_fw_rin",
                     "fw_shape.s")
        self.connect("radial_build.ib.FW R_out", "fw_shape.R_min")

        # build blanket shape
        self.add_subsystem("bl_shape",
                           PolarParallelCurve(use_Rmin=True, torus_V=True),
                           promotes_inputs=["R0"])
        self.connect("radial_build.ob_plasma_to_x.ob_pl_to_bl_rout",
                     "bl_shape.s")
        self.connect("radial_build.ib.blanket R_in", "bl_shape.R_min")

        # build exclusion zone of shield shape
        self.add_subsystem("exclusion_zone",
                           PolarParallelCurve(use_Rmin=True, torus_V=True),
                           promotes_inputs=["R0"])
        self.connect("radial_build.ob_plasma_to_x.ob_pl_to_sh_rout",
                     "exclusion_zone.s")
        self.connect("radial_build.ib.WC shield R_in", "exclusion_zone.R_min")

        # connect plasma shape to fw, bl, and exlcusion zone shapes
        self.connect("plasma.R",
                     ["fw_shape.R", "bl_shape.R", "exclusion_zone.R"])
        self.connect("plasma.Z",
                     ["fw_shape.Z", "bl_shape.Z", "exclusion_zone.Z"])
        self.connect(
            "plasma.dR_dθ",
            ["fw_shape.dR_dθ", "bl_shape.dR_dθ", "exclusion_zone.dR_dθ"])
        self.connect(
            "plasma.dZ_dθ",
            ["fw_shape.dZ_dθ", "bl_shape.dZ_dθ", "exclusion_zone.dZ_dθ"])

        # compute shield and blanket volumes
        self.add_subsystem(
            "sh_bl_volumes",
            om.ExecComp(
                [
                    "V_sh_mat = V_sh_enc - V_bl_enc",
                    "V_bl_mat = V_bl_enc - V_fw_enc"
                ],
                V_sh_mat={'units': 'm**3'},
                V_sh_enc={'units': 'm**3'},
                V_bl_mat={'units': 'm**3'},
                V_bl_enc={'units': 'm**3'},
                V_fw_enc={'units': 'm**3'},
            ))
        self.connect("exclusion_zone.V", "sh_bl_volumes.V_sh_enc")
        self.connect("bl_shape.V", "sh_bl_volumes.V_bl_enc")
        self.connect("fw_shape.V", "sh_bl_volumes.V_fw_enc")

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

        self.add_subsystem("cryostat", SimpleCryostat())
        self.connect("radial_build.cryostat R_out", "cryostat.R_out")
        self.connect("coils.half-height", "cryostat.TF half-height")

        self.add_subsystem(
            "divertor_area",
            om.ExecComp("Adiv = 0.1 * Awall",
                        Awall={'units': 'm**2'},
                        Adiv={'units': 'm**2'}))
        self.connect("plasma.S", "divertor_area.Awall")


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

        self.add_subsystem("magnet_quantities", TFMagnetSet())
        self.connect("geometry.radial_build.ib_tf.approximate cross section",
                     "magnet_quantities.cross section")
        self.connect("geometry.coils.arc length",
                     "magnet_quantities.arc length")
        self.connect("magnets.I_leg", "magnet_quantities.I_leg")

        self.add_subsystem("rf_heating", SimpleRFHeating(config=config))

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
        self.connect("rf_heating.P", ["plasma.P_heat.P_RF"])

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
        self.connect("rf_heating.P", ["pplant.P_RF"])
        self.connect("rf_heating.eff", ["pplant.η_RF"])
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
        self.connect("geometry.radial_build.ib.FW R_out", "q_n_IB.r_in")

        self.connect("geometry.plasma.S", "q_n.SA")
        self.connect("plasma.DTfusion.P_n", ["q_n_IB.P_n", "q_n.P_n"])
        self.connect("plasma.DTfusion.rate_fus", ["q_n_IB.S"])
        self.connect("q_n_IB.q_n", ["q_n.q_n_IB"])

        # compute magnet lifetime w.r.t. neutron damage
        self.add_subsystem("maglife", MenardMagnetLifetime(config=config))
        self.connect("q_n_IB.q_n", "maglife.q_n_IB")
        self.connect("geometry.blanket_sh.Shielding factor",
                     "maglife.Shielding factor")

        costing = GeneromakCosting()
        self.add_subsystem("costing", costing)
        self.connect("geometry.cryostat.V", "costing.V_FI")
        self.connect("magnet_quantities.V_set", "costing.V_pc")
        # # using the 'exact' generomak formulation the structure volume is
        # # estimated
        self.connect("geometry.sh_bl_volumes.V_sh_mat", "costing.V_sg")
        self.connect("geometry.sh_bl_volumes.V_bl_mat", "costing.V_bl")

        self.connect("geometry.divertor_area.Adiv", "costing.A_tt")
        self.connect("pplant.aux.P_aux,h", "costing.P_aux")
        self.connect("pplant.gen.P_el", "costing.P_e")
        self.connect("pplant.overall.P_net", "costing.P_net")
        self.connect("pplant.thermal.P_heat", "costing.P_t")
        self.connect("plasma.DTfusion.P_fus", "costing.P_fus")
        self.connect("q_n.q_n_avg", "costing.p_wn")
        self.connect("SOL.q_max", "costing.p_tt")
        costing.set_input_defaults("F_tt", 20, units="MW*a/m**2")
        costing.set_input_defaults("F_wn", 15, units="MW*a/m**2")
        costing.set_input_defaults("f_av", 0.6)


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator("sauter_generomak.yaml")
    machine = Machine(config=uc)
    prob.model = machine

    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output("a", units="m")
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

    prob.driver = om.pyOptSparseDriver()

    prob.driver.options['optimizer'] = 'IPOPT'
    prob.driver.opt_settings['print_level'] = 4

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
    prob.model.add_design_var('rf_heating.P',
                              lower=0,
                              upper=100,
                              ref=10,
                              units="MW")

    prob.model.add_objective('costing.COE', scaler=0.01)

    # set constraints
    prob.model.add_constraint('magnets.constraint_max_stress',
                              lower=0,
                              upper=0.05,
                              ref=0.1)
    prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0, upper=1)
    prob.model.add_constraint('magnets.constraint_wp_current_density',
                              lower=0,
                              ref=20)
    prob.model.add_constraint('R0', lower=3.0, upper=9.0, units='m')
    prob.model.add_constraint('geometry.aspect_ratio',
                              lower=1.8,
                              upper=3,
                              ref=1)
    prob.model.add_constraint('plasma.Hbalance.H', lower=0.99, upper=1.01)
    prob.model.add_constraint('geometry.adaptor.r_c', lower=0.2)
    prob.model.add_constraint('pplant.overall.P_net',
                              lower=599,
                              upper=601,
                              units='MW',
                              ref=500)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    # initial values for design variables
    prob.set_val("geometry.radial_build.CS ΔR", 0.45, units="m")
    prob.set_val("geometry.radial_build.ib_tf.Δr_s", 0.25, units='m')
    prob.set_val("geometry.radial_build.ib_tf.Δr_m", 0.25, units='m')
    prob.set_val('a', 1.7, units="m")
    prob.set_val("magnets.j_HTS", 70, units="MA/m**2")
    prob.set_val("plasma.NBIsource.P", 280, units="MW")
    prob.set_val('rf_heating.P', 0, units="MW")
    prob.set_val("geometry.adaptor.f_c", 0.9)
    prob.set_val("geometry.adaptor.Z_1", -2.0)

    # parameters
    δ = -0.7
    prob.set_val("geometry.plasma.δ", δ)
    prob.set_val("plasma.δ", δ)
    prob.set_val("SOL.δ", δ)

    n_coil = 18
    prob.set_val('geometry.radial_build.ib_tf.n_coil', n_coil)
    prob.set_val('magnet_quantities.n_coil', n_coil)
    prob.set_val('geometry.ripple.n_coil', n_coil)
    prob.set_val("costing.N_years", 25)
    prob.set_val("costing.T_constr", 5)

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
    newton.options['maxiter'] = 12
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
                                        print_arrays=False,
                                        units=True)
    all_outputs = prob.model.list_outputs(values=True,
                                          print_arrays=False,
                                          units=True)

    fig, ax = plt.subplots()
    machine.geometry.plasma.geom.plot(ax, label="Plasma")
    machine.geometry.fw_shape.limiter.plot(ax, label="First wall")
    machine.geometry.bl_shape.limiter.plot(ax, label="Blanket")
    machine.geometry.exclusion_zone.limiter.plot(ax, label="Shield")
    machine.geometry.coils.plot(ax, label="TF Coils (inner profile)")
    ax.set_xlim([-1, 8])
    ax.axis('equal')
    ax.legend()
    plt.show()
