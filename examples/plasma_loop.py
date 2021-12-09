# This is first example with a cycle of components, necessitating(?)
# a nonlinear solver for the loop.

from faroes.configurator import UserConfigurator
import faroes.units  # noqa: F401

from faroes.elliptical_plasma import MenardPlasmaGeometry
from faroes.menardplasmaloop import MenardPlasmaLoop

import openmdao.api as om


class Machine(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("plasmageom",
                           MenardPlasmaGeometry(config=config),
                           promotes_inputs=["R0", "A", "a"],
                           promotes_outputs=["ε", "κa", "V"])

        mpl = MenardPlasmaLoop(config=config)
        self.add_subsystem("plasma", mpl,
                           promotes_inputs=["R0", "Bt", "ε",
                                            ("minor_radius", "a"),
                                            "κa", "V",
                                            ("aspect_ratio", "A")])
        self.connect("plasmageom.L_pol", "plasma.L_pol")
        self.connect("plasmageom.L_pol_simple", "plasma.L_pol_simple")


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()
    prob.model = Machine(config=uc)

    model = prob.model

    prob.setup()

    prob.set_val('R0', 3, units='m')
    prob.set_val('A', 1.6)
    prob.set_val('a', 1.875)
    prob.set_val('Bt', 2.094, units='T')

    # initial inputs
    prob.set_val('plasma.Hbalance.H', 1.77)
    prob.set_val('plasma.Ip', 15.28, units="MA")
    prob.set_val("plasma.<n_e>", 1.107, units="n20")
    prob.set_val("plasma.radiation.rad.P_loss", 92.2, units="MW")

    mpl = prob.model.plasma
    newton = mpl.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 20
    mpl.linear_solver = om.DirectSolver()

    # initial values

    prob.run_driver()

    all_inputs = prob.model.list_inputs(val=True,
                                        print_arrays=True,
                                        units=True)
    all_outputs = prob.model.list_outputs(val=True,
                                          print_arrays=True,
                                          units=True)
