import openmdao.api as om
import faroes.util as util
from faroes.configurator import UserConfigurator


class ConfinementTimeScaling(om.ExplicitComponent):
    r"""Confinement time scaling law

    Options
    -------
    scaling : str
        The scaling law to use. The default is specified in fits.yaml.

    Inputs
    ------
    Ip : float
        MA, plasma current
    Bt : float
        T, toroidal field on axis
    n19 : float
        electron density in units of 10^19 per cubic meter
    PL : float
        MW, heating power (or loss power)
    R : float
        m, major radius
    ε : float
        inverse aspect ratio a / R
    κa : float
        effective elongation, S_c / (π a^2),
        where S_c is the plasma cross-sectional area
    M : float
        ion mass number

    Outputs
    -------
    τe : float
        s, confinement time

    Notes
    -----
    Exponents are loaded from fits.yaml. Not all inputs to this component are
    necessarily used.
    """
    BAD_TERM_STR = """Unknown term '%s' in confinement scaling.
    Valid terms are %s """

    def initialize(self):
        self.options.declare("config", default=None)
        self.options.declare("scaling", default=None)

    def setup(self):
        config = self.options["config"].accessor(["fits", "τe"])
        scaling = self.options["scaling"]
        if scaling is None:
            scaling = config(["default"])
        terms = config([scaling])

        valid_terms = ["c0", "Ip", "Bt", "n19", "PL", "R", "ε", "κa", "M"]
        for k, v in terms.items():
            if k not in valid_terms:
                raise ValueError(self.BAD_TERM_STR % (k, valid_terms))

        self.constant = terms.pop("c0")
        self.varterms = terms

        self.add_input("Ip", units="MA", desc="Plasma current")
        self.add_input("Bt", units="T", desc="Toroidal field on axis")
        self.add_input("n19", desc="Density in 10^19 m⁻³")
        self.add_input("PL", units="MW", desc="Heating power (or loss power)")
        self.add_input("R", units="m", desc="major radius")
        self.add_input("ε", desc="Inverse aspect ratio")
        self.add_input("κa", desc="Effective elongation, S_c / πa²")
        self.add_input("M", desc="Ion mass number")

        self.add_output("τe", units="s", desc="Energy confinement time")

    def compute(self, inputs, outputs):
        τe = self.constant
        for k, v in self.varterms.items():
            τe *= inputs[k]**v

        outputs["τe"] = τe

    def setup_partials(self):
        for k, v in self.varterms.items():
            self.declare_partials("τe", k)

    def partial(self, inputs, J, var):
        dτedv = self.constant
        for k, v in self.varterms.items():
            if k != var:
                dτedv *= inputs[k]**v
            else:
                dτedv *= v * inputs[k]**(v - 1)
        J["τe", var] = dτedv

    def compute_partials(self, inputs, J):
        for k, v in self.varterms.items():
            self.partial(inputs, J, k)


class MenardHybridScaling(om.Group):
    def initialize(self):
        self.options.declare('config')

    def setup(self):
        config = self.options['config']

        self.add_subsystem("nstxmg",
                           ConfinementTimeScaling(config=config,
                                                  scaling="NSTX-MG"),
                           promotes_inputs=['*'],
                           promotes_outputs=[("τe", "tau_N")])
        self.add_subsystem("petty",
                           ConfinementTimeScaling(config=config,
                                                  scaling="Petty"),
                           promotes_inputs=["*"],
                           promotes_outputs=[("τe", "tau_P")])

        self.add_subsystem('frac',
                           om.ExecComp("f = 0.5 + tanh((eps - 0.5) * 10) / 2"),
                           promotes=[("eps", "ε")])
        self.add_subsystem("tau",
                           om.ExecComp("tau = tau_N * f + (1-f) * tau_P",
                                       tau={"units": "s"},
                                       tau_N={"units": "s"},
                                       tau_P={"units": "s"}),
                           promotes=[("tau", "τ"), "tau_N", "tau_P"])
        self.connect("frac.f", "tau.f")


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model = MenardHybridScaling(config=uc)

    prob.setup()

    prob.set_val('Ip', 14.67, units='MA')
    prob.set_val('Bt', 2.094, units='T')
    prob.set_val('n19', 10.63)
    prob.set_val('PL', 83.34)
    prob.set_val('R', 3.0, units='m')
    prob.set_val('ε', 1 / 1.6)
    prob.set_val('κa', 2.19)
    prob.set_val('M', 2.5)
    prob.run_driver()
    prob.model.list_inputs(values=True)
    prob.model.list_outputs(values=True)
