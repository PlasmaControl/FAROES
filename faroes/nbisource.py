import openmdao.api as om
from plasmapy.particles import Particle
from faroes.configurator import UserConfigurator, Accessor
from astropy import units as apunits


class SimpleNBISourceProperties(om.ExplicitComponent):
    r"""Helper for the SimpleNBISource

    Notes
    -----
    A and Z are integers and won't change over the course of the simulation,
    but for (me as a naive-user) technical reasons, they can't easily be
    converted to being a discrete output.
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        acc = Accessor(self.options['config'])
        f = acc.accessor(["h_cd", "NBI"])
        acc.set_output(self, f, "energy", component_name="E", units="keV")
        acc.set_output(self, f, "power", component_name="P", units="MW")

        config = self.options["config"].accessor(["h_cd", "NBI"])
        species_name = config(['ion'])
        species = Particle(species_name)
        beam_ion_Z = species.integer_charge
        beam_ion_mass_number = species.mass_number
        beam_ion_mass = species.mass.to(apunits.kg).value
        self.add_output("m", units='kg', val=beam_ion_mass)
        self.add_output("A", val=beam_ion_mass_number)
        self.add_output("Z", val=beam_ion_Z)


class SimpleNBISource(om.Group):
    r"""Represents one or more neutral beam injectors

    Does not handle geometry (incidence angles) etc,
    or beams with multiple energy components.

    Notes
    -----
    A, Z will typically be fixed during a given set of simulation runs.

    Inputs
    ------
    P : float
        MW, Neutral beam power to plasma
    E : float
        keV, energy per particle
    A : float
        mass number of beam particles
    Z : int
        charge number of beam particles

    Outputs
    -------
    rate : float
        s^{-1}, particles per second
    v : float
        m/s, particle velocity
    """
    def initialize(self):
        self.options.declare("config", default=None)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           SimpleNBISourceProperties(config=config),
                           promotes_outputs=["P", "E", "A", "Z", "m"])
        self.add_subsystem(
            "flow",
            om.ExecComp("f = P/E",
                        f={'units': '1/s'},
                        P={'units': 'W'},
                        E={'units': 'J'}), promotes_outputs=["*"])
        self.add_subsystem(
            "v",
            om.ExecComp(
                "v = (2 * E/m)**(1/2)",
                v={'units': 'm/s'},
                E={'units': 'J'},
                m={'units': 'kg'},
            ), promotes_outputs=["*"])
        self.connect('E', ['flow.E', 'v.E'])
        self.connect('P', ['flow.P'])
        self.connect('m', ['v.m'])


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model = SimpleNBISource(config=uc)

    prob.setup()
    prob.run_driver()

    prob.model.list_inputs(values=True)
    prob.model.list_outputs(values=True)
