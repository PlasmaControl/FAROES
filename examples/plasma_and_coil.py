# This is an example of how to join components together

import openmdao.api as om
from faroes.simple_tf_magnet import MagnetRadialBuild
from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.configurator import UserConfigurator


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("plasma",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0"],
                           promotes_outputs=["R_max", "R_min"])
        self.add_subsystem('connector_ob',
                           om.ExecComp('r_iu = 2.5 + R_max',
                                       r_iu={'units': 'm'},
                                       R_max={'units': 'm'}),
                           promotes=['r_iu', 'R_max'])

        self.add_subsystem(
            'connector_ib',
            om.ExecComp('r_ot = R_min - 0.5',
                        r_ot={'units': 'm'},
                        R_min={'units': 'm'}))

        self.add_subsystem("magnets",
                           MagnetRadialBuild(config=config),
                           promotes_inputs=["R0", 'r_iu'])
        self.connect('R_min', ['connector_ib.R_min'])
        self.connect('connector_ib.r_ot', ['magnets.r_ot'])


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True

    prob.model.add_design_var('R0', lower=2.7, upper=3.3, ref=3.0, units='m')
    prob.model.add_design_var('magnets.r_is', lower=0.03, upper=1.0, ref=0.3)
    prob.model.add_design_var('magnets.r_im', lower=0.05, upper=1.0, ref=0.3)
    prob.model.add_design_var('magnets.j_HTS', lower=10, upper=300, ref=100)

    prob.model.add_objective('magnets.obj')

    # set constraints
    prob.model.add_constraint('magnets.constraint_max_stress', lower=0)
    prob.model.add_constraint('magnets.constraint_B_on_coil', lower=0)
    prob.model.add_constraint('magnets.constraint_wp_current_density', lower=0)
    prob.model.add_constraint('magnets.r_im_is_constraint', lower=0)
    prob.model.add_constraint('magnets.A_s', lower=0)
    prob.model.add_constraint('magnets.A_m', lower=0)
    prob.model.add_constraint('magnets.A_t', lower=0)

    prob.setup()
    prob.check_config(checks=['unconnected_inputs'])

    prob.set_val('plasma.A', 2)
    prob.set_val('magnets.n_coil', 18)
    prob.set_val('magnets.windingpack.j_eff_max', 160)
    prob.set_val('magnets.windingpack.f_HTS', 0.76)
    prob.set_val("magnets.magnetstructure_props.Young's modulus", 220)

    prob.run_driver()

    all_inputs = prob.model.list_inputs(values=True)
    all_outputs = prob.model.list_outputs(values=True)
