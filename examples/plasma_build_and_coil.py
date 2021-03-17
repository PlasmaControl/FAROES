# This is an example of how to join components together

import openmdao.api as om
from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.radialbuild import MenardSTRadialBuild
from faroes.configurator import UserConfigurator


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("radial_build",
                           MenardSTRadialBuild(config=config),
                           promotes_inputs=["a", "CS R_out", "Ib TF R_out"],
                           promotes_outputs=["A", "R0", "Ib TF R_in"])

        self.add_subsystem("magnets",
                           MagnetRadialBuild(config=config),
                           promotes_inputs=["R0", ("r_is", "Ib TF R_in")],
                           promotes_outputs=["Ib TF R_out"])

        self.add_subsystem("plasmageom",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", "A", "a"])

        self.connect('radial_build.Ob TF R_min', ['magnets.r_iu'])

        self.connect('magnets.Ob TF R_out', ['radial_build.Ob TF R_out'])


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['disp'] = True

    prob.model.add_design_var('CS R_out',
                              lower=0.03,
                              upper=2.0,
                              ref=0.2,
                              units='m')
    prob.model.add_design_var('a', lower=0.1, upper=2, units='m')
    prob.model.add_design_var('magnets.Δr_s', lower=0.05, upper=1.5, ref=0.3)
    prob.model.add_design_var('magnets.Δr_m', lower=0.05, upper=1.5, ref=0.3)
    prob.model.add_design_var('magnets.j_HTS', lower=10, upper=250, ref=100)

    prob.model.add_objective('magnets.B0', scaler=-1)

    # set constraints
    prob.model.add_constraint('magnets.constraint_max_stress', lower=0)
    prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0)
    prob.model.add_constraint('magnets.constraint_wp_current_density', lower=0)
    prob.model.add_constraint('R0', equals=3.0)
    prob.model.add_constraint('A', equals=3.0)

    prob.setup()

    prob.set_val("CS R_out", 0.30)
    prob.set_val("magnets.Δr_s", 0.30, units='m')
    prob.set_val("magnets.Δr_m", 0.30, units='m')
    prob.set_val("a", 1.00, units='m')

    prob.set_val('magnets.n_coil', 18)
    prob.set_val('magnets.windingpack.j_eff_max', 160)
    prob.set_val('magnets.windingpack.f_HTS', 0.76)
    prob.set_val("magnets.magnetstructure_props.Young's modulus", 220)

    model = prob.model
    newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    model.linear_solver = om.DirectSolver()

    prob.run_driver()

    # all_inputs = prob.model.list_inputs(values=True)
    # all_outputs = prob.model.list_outputs(values=True)
