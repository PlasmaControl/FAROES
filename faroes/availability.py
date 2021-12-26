import openmdao.api as om
from faroes.configurator import UserConfigurator


class AvailabilityProperties(om.Group):
    BAD_BL_MODEL = "Availability model %s not supported." \
        "Choice is 'divertorsOnly'."

    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        if self.options['config'] is None:
            raise ValueError("Configuration tree required.")
        config = self.options["config"]
        f = config.accessor(["machine", "availability"])
        model = f(["model"])
        if model == "divertorsOnly":
            f_tt = f([model, "lifetime"], units="MW*a/m**2")
            t_replace = f([model, "replacement time"], units="a")
            ivc = om.IndepVarComp()
            ivc.add_output("F_tt",
                           val=f_tt,
                           units="MW*a/m**2",
                           desc="Divertor durability")
            ivc.add_output("divertors t_replace",
                           val=t_replace,
                           units="a",
                           desc="Maintenance time for divertor replacement")
            self.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        else:
            raise ValueError(self.BAD_BL_MODEL % (model))


class DivertorsOnlyAvailability(om.ExplicitComponent):
    r"""Simple availability analysis based on divertor replacement.

    Ignores blanket replacement. (One might imagine that it occurs
    in parallel with the divertor replacement.)

    Inputs
    ------
    F_tt : float
        MW*a/m**2, Divertor lifetime against neutrons, erosion, etc.
    p_tt : float
        MW/m**2, Maximum divertor heatflux; determines useful life
    t_replace: float
        d, Time to replace all the divertors

    Outputs
    -------
    f_av : float
        Availability
    time between repl
        a, time between replacements

    References
    ----------
    .. [1] Nagy, D.; Bonnemason, J. DEMO Divertor Maintenance.
    Fusion Engineering and Design 2009, 84 (7–11), 1388–1393.
    https://doi.org/10.1016/j.fusengdes.2009.01.102.
    """
    def setup(self):
        self.add_input("F_tt", units="MW*a/m**2", desc="Divertor durability")
        self.add_input("p_tt", units="MW/m**2", desc="Peak divertor heat flux")
        self.add_input("t_replace",
                       units='a',
                       desc="Maintenance time for divertor replacement")
        self.add_output("f_av", desc="Availability factor")
        self.add_output("time between replacements", units='a')

    def compute(self, inputs, outputs):
        ftt = inputs["F_tt"]
        ptt = inputs["p_tt"]
        t_replace = inputs["t_replace"]
        time_betw_repl = ftt / ptt
        outputs["time between replacements"] = time_betw_repl

        f_av = time_betw_repl / (time_betw_repl + t_replace)
        outputs["f_av"] = f_av

    def setup_partials(self):
        self.declare_partials("f_av", ["F_tt", "p_tt", "t_replace"])
        self.declare_partials("time between replacements", ["F_tt", "p_tt"])

    def compute_partials(self, inputs, J):
        ftt = inputs["F_tt"]
        ptt = inputs["p_tt"]
        t_replace = inputs["t_replace"]
        numer = ptt * t_replace
        denom = (ptt * t_replace + ftt)**2
        J["f_av", "F_tt"] = numer / denom
        numer = -(ftt * t_replace)
        J["f_av", "p_tt"] = numer / denom
        numer = -(ftt * ptt)
        J["f_av", "t_replace"] = numer / denom
        J["time between replacements", "F_tt"] = 1 / ptt
        J["time between replacements", "p_tt"] = -ftt / ptt**2


class SimpleAvailability(om.Group):
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)

    def setup(self):
        config = self.options['config']
        if config is None:
            raise ValueError("Configuration tree required.")
        self.add_subsystem("props",
                           AvailabilityProperties(config=config),
                           promotes=["*"])
        self.add_subsystem("av",
                           DivertorsOnlyAvailability(),
                           promotes_inputs=[("t_replace",
                                             "divertors t_replace"), "F_tt",
                                            "p_tt"],
                           promotes_outputs=["f_av"])


if __name__ == "__main__":
    prob = om.Problem()
    uc = UserConfigurator()

    prob.model = SimpleAvailability(config=uc)
    prob.setup(force_alloc_complex=True)
    prob.set_val("p_tt", 10, units="MW/m**2")
    prob.run_driver()
    all_inputs = prob.model.list_inputs(val=True, units=True, desc=True)
    all_outputs = prob.model.list_outputs(val=True, units=True, desc=True)
