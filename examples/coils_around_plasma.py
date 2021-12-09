# This is an example of how to join components together

import openmdao.api as om
from faroes.sauter_plasma import SauterGeometry
from faroes.threearcdeecoil import ThreeArcDeeTFSet, ThreeArcDeeTFSetAdaptor
# from faroes.princetondeecoil import PrincetonDeeTFSet
from faroes.configurator import UserConfigurator

from faroes.util import PolarParallelCurve

import numpy as np
from scipy.constants import pi

import matplotlib.pyplot as plt


class Machine(om.Group):
    def initialize(self):
        self.options.declare("config")

    def setup(self):
        config = self.options["config"]

        self.add_subsystem("plasma",
                           SauterGeometry(config=config),
                           promotes_inputs=["R0", ("θ", "θ_for_d2")],
                           promotes_outputs=["R_out", "R_in"])

        self.add_subsystem("exclusion_zone",
                           PolarParallelCurve(use_Rmin=True),
                           promotes_inputs=["R0", ("s", "offset")])
        self.connect("plasma.R", "exclusion_zone.R")
        self.connect("plasma.Z", "exclusion_zone.Z")
        self.connect("plasma.dR_dθ", "exclusion_zone.dR_dθ")
        self.connect("plasma.dZ_dθ", "exclusion_zone.dZ_dθ")

        self.add_subsystem("ObRin",
                           om.ExecComp("R_in = R0 + dR",
                                       dR={'units': 'm'},
                                       R_in={'units': 'm'},
                                       R0={'units': 'm'}),
                           promotes_inputs=["R0"])

        self.add_subsystem("adaptor",
                           ThreeArcDeeTFSetAdaptor(),
                           promotes_inputs=["Ib TF R_out"])
        self.add_subsystem("coils",
                           ThreeArcDeeTFSet(),
                           promotes_inputs=["R0", "Ib TF R_out"],
                           promotes_outputs=["V_enc"])

        # I really should connect a value representing the minimum height of
        # the offset shape, but it should be alright this way too.
        self.connect("plasma.b", "adaptor.Z_min")
        self.connect("adaptor.hhs", "coils.hhs")
        self.connect("adaptor.e_a", "coils.e_a")
        self.connect("adaptor.r_c", "coils.r_c")

        # self.add_subsystem("coils",
        #                   PrincetonDeeTFSet(),
        #                   promotes_inputs=["R0"],
        #                   promotes_outputs=["V_enc"])

        self.connect("exclusion_zone.θ_parall", "coils.θ")
        self.connect("ObRin.R_in", "adaptor.Ob TF R_in")

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
            ))

        self.connect("coils.d_sq", "margin.a")
        self.connect("exclusion_zone.d_sq", "margin.b")

        self.add_subsystem(
            'ks',
            om.KSComp(width=100,
                      units="m**2",
                      ref=10,
                      lower_flag=True,
                      rho=10,
                      upper=0,
                      add_constraint=True))
        self.connect("margin.c", "ks.g")


if __name__ == "__main__":
    prob = om.Problem()

    θ = np.linspace(0, 2 * pi, 100, endpoint=False)
    prob.model.add_subsystem("ivc",
                             om.IndepVarComp("θ_for_d2", val=θ),
                             promotes_outputs=["*"])

    uc = UserConfigurator()
    machine = Machine(config=uc)
    prob.model.add_subsystem("machine", machine, promotes_inputs=["*"])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["disp"] = True

    # Design variables for the ThreeArcDee coil
    prob.model.add_design_var("Ib TF R_out", lower=1.0, upper=4.5, units="m")
    prob.model.add_design_var("adaptor.f_c", lower=0.01, upper=0.99)
    prob.model.add_design_var("adaptor.Z_1", lower=0.0, upper=5.0, units="m")

    # prob.model.add_design_var("coils.Ib TF R_out",
    #                           lower=1.0,
    #                           upper=4.5,
    #                           ref=1.5,
    #                           units="m")
    prob.model.add_design_var("ObRin.dR",
                              lower=0.5,
                              upper=25.0,
                              ref=3.0,
                              units="m")

    prob.model.add_objective("machine.V_enc")

    prob.setup(force_alloc_complex=True)

    # set the plasma shape
    prob.set_val("R0", 6.0, units="m")
    prob.set_val("plasma.A", 2.4)
    prob.set_val("plasma.a", 2.5)
    prob.set_val("plasma.κ", 2.5)
    prob.set_val("plasma.δ", -0.4)
    prob.set_val("plasma.ξ", 0.05)
    prob.set_val("offset", 2.0)
    prob.set_val("exclusion_zone.R_min", 2.5)

    # initial values for the design variables
    # since we are 'shrink-wrapping' the coils, it's good to initially make
    # them very large.

    prob.set_val('adaptor.f_c', 0.4)
    prob.set_val('adaptor.Z_1', 5.0, units='m')
    prob.set_val("ObRin.dR", 15.0, units="m")

    prob.run_driver()

    all_inputs = prob.model.list_inputs(val=True, print_arrays=True)
    all_outputs = prob.model.list_outputs(val=True, print_arrays=True)

    blanket_d_sq = prob.get_val("machine.exclusion_zone.d_sq")
    blanket_theta = prob.get_val("machine.exclusion_zone.θ_parall")
    blanket_d = np.sqrt(blanket_d_sq)
    blanket_R = prob.get_val("R0") + blanket_d * np.cos(blanket_theta)
    blanket_Z = blanket_d * np.sin(blanket_theta)

    fig, ax = plt.subplots()
    machine.plasma.plot(ax)
    machine.coils.plot(ax)
    ax.plot(blanket_R, blanket_Z, marker="x")
    ax.set_xlim([-1, 8])
    ax.axis('equal')
    plt.show()
