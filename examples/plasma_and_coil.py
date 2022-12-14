# This is an example of how to join components together

import openmdao.api as om
from faroes.simple_tf_magnet import ExampleMagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.configurator import UserConfigurator


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("aspect",
                           om.ExecComp('A = R0/a',
                                       R0={'units': 'm'},
                                       a={'units': 'm'}),
                           promotes_inputs=['R0', 'a'])

        self.add_subsystem("magnets",
                           ExampleMagnetRadialBuild(config=config),
                           promotes_inputs=['r_iu', 'R0'])

        self.add_subsystem('connector_ib',
                           om.ExecComp('R0 = r_ot + 0.5 + a',
                                       r_ot={'units': 'm'},
                                       R0={'units': 'm'},
                                       a={'units': 'm'}),
                           promotes_inputs=["a"],
                           promotes_outputs=["R0"])
        self.connect('magnets.Ib TF R_out', 'connector_ib.r_ot')

        self.add_subsystem("plasma",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", "a"],
                           promotes_outputs=["R_out"])
        self.connect('aspect.A', 'plasma.A')


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True

    prob.model.add_design_var('a', lower=0.3, upper=3, ref=1)
    prob.model.add_design_var('magnets.r_is', lower=0.03, upper=0.3, ref=0.3)
    prob.model.add_design_var('magnets.Δr_s', lower=0.05, upper=0.95, ref=0.5)
    prob.model.add_design_var('magnets.Δr_m', lower=0.05, upper=0.95, ref=0.5)
    prob.model.add_design_var('magnets.eng.j_HTS',
                              lower=10,
                              upper=300,
                              ref=100)

    prob.model.add_objective('magnets.B0', scaler=-1)

    # set constraints
    prob.model.add_constraint('magnets.eng.constraint_max_stress', lower=0)
    prob.model.add_constraint('magnets.eng.constraint_B_on_coil', lower=0)
    prob.model.add_constraint('magnets.eng.constraint_wp_current_density',
                              lower=0)
    prob.model.add_constraint('aspect.A', equals=2.0)
    prob.model.add_constraint('R0', equals=3.5)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val("magnets.r_is", 0.2, units="m")
    prob.set_val("magnets.Δr_s", 0.2, units="m")
    prob.set_val("magnets.Δr_m", 0.2, units="m")
    prob.set_val("magnets.ob_gap.r_min", 8.045, units="m")
    prob.set_val("plasma.a", 0.5, units="m")
    prob.set_val("aspect.A", 2.0)
    prob.set_val('magnets.n_coil', 18)
    prob.set_val('magnets.eng.windingpack.j_eff_max', 160)
    prob.set_val('magnets.eng.windingpack.f_HTS', 0.76)
    prob.set_val("magnets.eng.magnetstructure_props.Young's modulus", 220)

    model = prob.model
    newton = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    model.linear_solver = om.DirectSolver()

    prob.run_driver()

    all_inputs = prob.model.list_inputs(val=True)
    all_outputs = prob.model.list_outputs(val=True)
