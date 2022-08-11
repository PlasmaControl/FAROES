import faroes.units  # noqa: F401
from faroes.configurator import UserConfigurator

import openmdao.api as om


class ConfinementTime(om.Group):
    r"""Scaled energy confinement time

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.
    scaling : str
        The scaling law to use. If ``"default"`` or ``None``, uses the
        "default" option specified in ``fits.yaml``

    Inputs
    ------
    H : float
        H-factor; multiple of the confinement time compared to that
        expected from the scaling law
    Ip : float
        MA, plasma current
    Bt : float
        T, toroidal field on axis
    n19 : float
        n19, electron density
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
        u, main ion mass number

    Outputs
    -------
    τe : float
        s, confinement time

    """
    def initialize(self):
        self.options.declare('config', default=None, recordable=False)
        self.options.declare("scaling", default=None)

    def setup(self):
        config = self.options['config']
        scaling = self.options['scaling']

        # a bit of a special case; could be done better perhaps
        if scaling == "MenardHybrid":
            self.add_subsystem("law",
                               MenardHybridScaling(config=config),
                               promotes_inputs=["*"])
        else:
            self.add_subsystem("law",
                               ConfinementTimeScaling(config=config,
                                                      scaling=scaling),
                               promotes_inputs=["*"])
        self.add_subsystem("withH",
                           ConfinementTimeMultiplication(),
                           promotes_inputs=["H"],
                           promotes_outputs=["τe"])
        self.connect("law.τe", ["withH.τe_law"])


class ConfinementTimeMultiplication(om.ExplicitComponent):
    r"""Energy confinement time

    .. math:: \tau_e = H \tau_{e,\mathrm{law}}

    Inputs
    ------
    τe_law : float
        s, confinement time as determined by scaling law
    H : float
        H-factor; multiple of the confinement time

    Outputs
    -------
    τe : float
        s, confinement time

    Notes
    -----
    This is a component in order to allow use of greek variables; otherwise it
    would be fine as an ExecComp.
    """
    def setup(self):
        self.add_input("τe_law",
                       units="s",
                       desc="Confinement time as determined by scaling law")
        self.add_input("H",
                       val=1,
                       desc="H-factor; multiplies confinement time")
        self.add_output("τe", units="s", desc="Energy confinement time")

    def compute(self, inputs, outputs):
        τe = inputs["H"] * inputs["τe_law"]
        outputs["τe"] = τe

    def setup_partials(self):
        self.declare_partials("τe", ["H", "τe_law"])

    def compute_partials(self, inputs, J):
        J["τe", "H"] = inputs["τe_law"]
        J["τe", "τe_law"] = inputs["H"]


class ConfinementTimeScaling(om.ExplicitComponent):
    r"""Confinement time scaling law

    .. math::
        \tau_e = c_0 I_p^{c_{I_p}}\,B_t^{c_{B_t}}\,n_{19}^{c_{n_{19}}}\,
                 P_L^{c_{P_L}}\, R^{c_R}\,\epsilon^{c_\epsilon}\,
                 \kappa_a^{c_{\kappa_a}}\,M^{c_M}


    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.
    scaling : str
        The scaling law to use. If ``"default"`` or ``None``, uses the
        "default" option specified in ``fits.yaml``.

    Inputs
    ------
    Ip : float
        MA, plasma current
    Bt : float
        T, toroidal field on axis
    n19 : float
        n19, electron density
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
        main ion mass number

    Outputs
    -------
    τe : float
        s, confinement time

    Raises
    ------
    ValueError
        If any key in the scaling law dictionary is not one of a set of known
        values.
    om.AnalysisError
        If any of the input values is negative during a computation.

    Notes
    -----
    Exponents are loaded from fits.yaml. Not all inputs to this component are
    necessarily used.
    """
    BAD_TERM = """Unknown term '%s' in confinement scaling.
    Valid terms are %s """

    NEGATIVE_TERM = "Term '%s' is non-positive in " + \
        "the confinement time calculation. Its value was %f."

    def initialize(self):
        self.options.declare("config", default=None, recordable=False)
        self.options.declare("scaling", default=None)

    def setup(self):
        config = self.options["config"].accessor(["fits", "τe"])
        scaling = self.options["scaling"]
        if scaling is None or scaling == "default":
            scaling = config(["default"])
        terms = config([scaling]).copy()

        valid_terms = ["c0", "Ip", "Bt", "n19", "PL", "R", "ε", "κa", "M"]
        for k, v in terms.items():
            if k not in valid_terms:
                raise ValueError(self.BAD_TERM % (k, valid_terms))

        self.constant = terms.pop("c0")
        self.varterms = terms

        self.add_input("Ip", units="MA", desc="Plasma current")
        self.add_input("Bt", units="T", desc="Toroidal field on axis")
        self.add_input("n19", units="n19", desc="Density")
        self.add_input("PL", units="MW", desc="Heating power (or loss power)")
        self.add_input("R", units="m", desc="Major radius")
        self.add_input("ε", desc="Inverse aspect ratio")
        self.add_input("κa", desc="Effective elongation, S_c / πa²")
        self.add_input("M", units="u", desc="Ion mass number")

        self.add_output("τe",
                        units="s",
                        lower=1e-3,
                        desc="Energy confinement time")

    def compute(self, inputs, outputs):
        τe = self.constant

        Bt = inputs["Bt"]
        if Bt < 0:
            raise om.AnalysisError(f"Magnetic field strength negative, {Bt} T")

        Ip = inputs["Ip"]
        if Ip > 1000:
            raise om.AnalysisError(f"Plasma current is too large, {Ip} MA")

        n19 = inputs["n19"]
        if n19 > 10000:
            raise om.AnalysisError(f"Density is too large, {n19} x e19")

        for k, v in self.varterms.items():
            term = inputs[k]
            if term <= 0:
                raise om.AnalysisError(self.NEGATIVE_TERM % (k, term))
            τe *= term**v

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
    r"""Aspect-ratio-dependent hybrid scaling law

    This law is indended to bridge the gap between spherical tokamaks
    and standard-aspect-ratio tokamaks, especially for :math:`A` of
    around 1.5 to 2.5.

    It uses a hyperbolic tangent to interpolate between the NSTX-MG
    scaling law at small aspect ratio and the Petty scaling
    :footcite:p:`petty_sizing_2008` at larger aspect ratio.

    .. math::

       \tau = f \tau_{e,\mathrm{NSTX-MG}} + (1 - f) \tau_{e,\mathrm{Petty}}

       f = 0.5 + \tanh(10 \epsilon - 5) / 2

    The interpolator :math:`f` is 0.5 at :math:`\epsilon=0.5` or :math:`A=2`.

    Options
    -------
    config : UserConfigurator
        Configuration tree. Required option.

    Inputs
    ------
    Ip : float
        MA, plasma current
    Bt : float
        T, toroidal field on axis
    n19 : float
        n19, electron density
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
        main ion mass number

    Outputs
    -------
    τe : float
        s, confinement time
    """
    def initialize(self):
        self.options.declare('config', recordable=False)

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

        self.add_subsystem(
            'frac',
            om.ExecComp(
                "f = 0.5 + tanh((eps - 0.5) * 10) / 2",
                eps={'desc': "Inverse aspect ratio"},
                f={'desc': "Inter-formula interpolation parameter"},
            ),
            promotes=[("eps", "ε")])
        self.add_subsystem(
            "tau",
            om.ExecComp("tau = tau_N * f + (1-f) * tau_P",
                        f={'desc': "Inter-formula interpolation parameter"},
                        tau={
                            'units': 's',
                            'desc': "Synthesized confinement time"
                        },
                        tau_N={
                            'units': 's',
                            'desc': "NSTX-MG scaling conf. time"
                        },
                        tau_P={
                            'units': 's',
                            'desc': "Petty scaling conf. time"
                        }),
            promotes=[("tau", "τe"), "tau_N", "tau_P"])
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
    prob.model.list_inputs(val=True, desc=True)
    prob.model.list_outputs(val=True, desc=True)
