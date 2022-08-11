import openmdao.api as om
from plasmapy.particles import Particle
from faroes.configurator import UserConfigurator, Accessor
from astropy import units as apunits


class SimpleNBISourceProperties(om.Group):
    r"""Helper for the SimpleNBISource

    Loads from the configuration tree::

        h_cd:
          NBI:
            power: <P>
            energy: <E>
            ion: <ion name>
            wall-plug efficiency: <eff>

    The ion name is parsed by ``plasmapy`` to determine A, Z, and m.
    Examples include ``D+``.

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Outputs
    -------
    P : float
        MW, Neutral beam power to plasma
    E : float
        keV, energy per particle
    A : float
        u, mass of beam particles
    Z : int
        Fundamental charges, Charge of beam particles
    m : float
        kg, Mass of beam particles
    eff : float
        Wall-plug efficiency

    Notes
    -----
    A and Z are (nearly) integers and won't change over
    the course of the simulation, but for technical reasons,
    they can't easily be converted to being a discrete output.
    """
    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

    def setup(self):
        ivc = om.IndepVarComp()
        acc = Accessor(self.options['config'])
        f = acc.accessor(["h_cd", "NBI"])
        acc.set_output(ivc, f, "energy", component_name="E", units="keV")
        acc.set_output(ivc, f, "power", component_name="P", units="MW")
        acc.set_output(ivc, f, "wall-plug efficiency", component_name="eff")

        config = self.options["config"].accessor(["h_cd", "NBI"])
        species_name = config(['ion'])
        species = Particle(species_name)
        beam_ion_Z = species.charge_number
        beam_ion_mass_number = species.mass_number
        beam_ion_mass = species.mass.to(apunits.kg).value
        ivc.add_output("m", units='kg', val=beam_ion_mass)
        ivc.add_output("A", units="u", val=beam_ion_mass_number)
        ivc.add_output("Z", val=beam_ion_Z)
        self.add_subsystem("ivc", ivc, promotes=["*"])


class SimpleNBISource(om.Group):
    r"""Represents one or more neutral beam injectors

    Does not handle geometry (incidence angles) etc,
    or beams with multiple energy components.

    The neutral beam power, initial energy, ion species,
    are efficiency are loaded using :class:`.SimpleNBISourceProperties`.

    The particle source rate is

    .. math:: S = P / E,

    the particle velocity is computed using

    .. math:: v = \sqrt{2 E/m},

    and the total wall-plug power required is

    .. math:: P_\mathrm{aux} = P / \mathrm{eff}.

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Outputs
    -------
    P : float
        MW, Neutral beam power to plasma
    E : float
        keV, energy per particle
    A : float
        u, mass of beam particles
    Z : int
        e, Charge of beam particles
    m : float
        kg, Mass of beam particles
    S : float
        s^{-1}, particles per second
    v : float
        m/s, particle velocity
    eff : float
        Wall-plug efficiency
    P_aux : float
        MW, Wall-plug power
    """
    def initialize(self):
        self.options.declare("config", default=None, recordable=False)

    def setup(self):
        config = self.options["config"]
        self.add_subsystem("props",
                           SimpleNBISourceProperties(config=config),
                           promotes_outputs=["P", "E", "A", "Z", "m", "eff"])
        self.add_subsystem("P_aux",
                           om.ExecComp(
                               "P_aux = P / eff",
                               P={
                                   "units": "MW",
                                   'desc': "NBI heating power"
                               },
                               eff={'desc': "NBI wall-plug efficiency"},
                               P_aux={
                                   "units": "MW",
                                   'desc': "NBI wall-plug power"
                               }),
                           promotes_inputs=["P", "eff"],
                           promotes_outputs=["P_aux"])
        self.add_subsystem("rate",
                           om.ExecComp("S = P/E",
                                       S={
                                           'units': '1/s',
                                           'desc': "Particle source rate"
                                       },
                                       P={
                                           'units': 'W',
                                           'desc': "Particle power"
                                       },
                                       E={
                                           'units': 'J',
                                           'desc': "Energy per particle"
                                       }),
                           promotes_outputs=["*"])
        self.add_subsystem("v",
                           om.ExecComp(
                               "v = (2 * E/m)**(1/2)",
                               v={
                                   'units': 'm/s',
                                   'desc': "Beam particle initial velocity"
                               },
                               E={
                                   'units': 'J',
                                   'desc': "Beam particle initial energy"
                               },
                               m={
                                   'units': 'kg',
                                   'desc': "Beam particle mass"
                               },
                           ),
                           promotes_outputs=["*"])
        self.connect('E', ['rate.E', 'v.E'])
        self.connect('P', ['rate.P'])
        self.connect('m', ['v.m'])


if __name__ == "__main__":
    prob = om.Problem()

    uc = UserConfigurator()

    prob.model = SimpleNBISource(config=uc)

    prob.setup()
    prob.run_driver()

    prob.model.list_inputs(val=True, desc=True, units=True)
    prob.model.list_outputs(val=True, desc=True, units=True)
