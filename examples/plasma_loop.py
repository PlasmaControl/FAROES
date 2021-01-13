# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver. However, it should be noted that here the there's not
# _actually_ a cycle. The only variables which cycle back are ZeroDPlasma.A to
# confinementtime.M (the average main ion mass) and NBIslowing.Wfast, the NBI
# fast particle energy. The latter does not play a role in computing p_th or
# the other main plasma properties.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

# from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import PlasmaGeometry
# from faroes.radialbuild import MenardSTRadialBuild
from faroes.simple_plasma import ZeroDPlasma
from faroes.nbisource import SimpleNBISource
from faroes.fastparticleslowing import FastParticleSlowing

from faroes.confinementtime import ConfinementTime

import openmdao.api as om


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("plasmageom",
                           PlasmaGeometry(config=config),
                           promotes_inputs=["R0"],
                           promotes_outputs=["ε", "κa"])

        self.add_subsystem("confinementtime",
                           ConfinementTime(config=config, scaling="default"),
                           promotes_inputs=[("R", "R0"), "ε", "κa", "Bt",
                                            ("n19", "<n_e>")])

        self.add_subsystem("ZeroDPlasma",
                           ZeroDPlasma(config=config),
                           promotes_inputs=["<n_e>"],
                           promotes_outputs=["<T_e>"])

        self.connect("plasmageom.V", ["ZeroDPlasma.V"])
        self.connect("confinementtime.τe", ["ZeroDPlasma.τ_th"])

        # back-connections
        self.connect("ZeroDPlasma.A", ["confinementtime.M"])

        self.add_subsystem("NBIsource", SimpleNBISource(config=config))
        self.add_subsystem("NBIslowing",
                           FastParticleSlowing(),
                           promotes_inputs=[("ne", "<n_e>"), ("Te", "<T_e>")])
        self.connect("ZeroDPlasma.ni", ["NBIslowing.ni"])
        self.connect("ZeroDPlasma.Ai", ["NBIslowing.Ai"])
        self.connect("ZeroDPlasma.Zi", ["NBIslowing.Zi"])

        # back-connections
        self.connect("NBIslowing.Wfast", ["ZeroDPlasma.W_fast_NBI"])

        self.connect("NBIsource.S", ["NBIslowing.S"])
        self.connect("NBIsource.E", ["NBIslowing.Wt"])
        self.connect("NBIsource.A", ["NBIslowing.At"])
        self.connect("NBIsource.Z", ["NBIslowing.Zt"])


#        self.add_subsystem(""

#        self.add_subsystem("radial_build",
#                           MenardSTRadialBuild(config=config),
#                           promotes_inputs=['CS R_max'])
#
#        self.add_subsystem("magnets",
#                           MagnetRadialBuild(config=config),
#                           promotes_inputs=["R0"])
#        self.connect('plasma.R_max', ['radial_build.plasma R_max'])
#        self.connect('plasma.R_min', ['radial_build.plasma R_min'])
#
#        self.connect('radial_build.Ob TF R_min', ['magnets.r_iu'])
#        self.connect('radial_build.Ib TF R_min', ['magnets.r_is'])
#        self.connect('radial_build.Ib TF R_max', ['magnets.r_ot'])
#
#        self.connect('magnets.Ob TF R_out', ['radial_build.Ob TF R_out'])

if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    # prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['disp'] = True

    # prob.model.add_design_var('plasma.A', lower=1.6, upper=4.0, ref=2.0)
    # prob.model.add_design_var('CS R_max',
    #                              lower=0.02,
    #                              upper=1.0,
    #                              ref=0.3,
    #                              units='m')
    # prob.model.add_design_var('magnets.r_im', lower=0.05, upper=1.0, ref=0.3)
    # prob.model.add_design_var('magnets.j_HTS', lower=10, upper=300, ref=100)

    # prob.model.add_objective('magnets.obj')

    # set constraints
    # prob.model.add_constraint('magnets.constraint_max_stress', lower=0)
    # prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0)
    # prob.model.add_constraint('magnets.constraint_wp_current_density',
    #                            lower=0)
    # prob.model.add_constraint('magnets.r_im_is_constraint', lower=0)
    # prob.model.add_constraint('magnets.A_s', lower=0)
    # prob.model.add_constraint('magnets.A_m', lower=0)
    # prob.model.add_constraint('magnets.A_t', lower=0)

    prob.model.set_input_defaults("<n_e>", val=1.0e20, units="m**-3")
    prob.setup()
    # prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('R0', 3, units='m')
    prob.set_val('plasmageom.A', 1.6)
    prob.set_val("<n_e>", 1.06, units="n20")
    prob.set_val('Bt', 2.094, units='T')

    # confinement time inputs
    prob.set_val('confinementtime.Ip', 14.67, units="MA")
    prob.set_val('confinementtime.PL', 83.34, units="MW")
    prob.set_val('confinementtime.H', 1.77)

    # plasma inputs
    prob.set_val("ZeroDPlasma.P_loss", 83.34, units="MW")
    prob.set_val("ZeroDPlasma.W_fast_α", 13.05, units="MJ")

    prob.set_val("NBIslowing.logΛe", 17.37)

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    # prob.model.nonlinear_solver = om.NonlinearBlockGS()
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.linear_solver = om.DirectSolver()

    prob.run_model()
    # prob.check_totals(of=['NBIslowing.Wfast'], wrt=['<n_e>'])

    #    prob.set_val('magnets.n_coil', 18)
    #    prob.set_val('magnets.windingpack.j_eff_max', 160)
    #    prob.set_val('magnets.windingpack.f_HTS', 0.76)
    #    prob.set_val("magnets.magnetstructure_props.Young's modulus", 220)

    prob.run_driver()
    prob.check_totals(of=['NBIslowing.Wfast'], wrt=['<n_e>'])

    all_inputs = prob.model.list_inputs(values=True, print_arrays=True)
    all_outputs = prob.model.list_outputs(values=True)
