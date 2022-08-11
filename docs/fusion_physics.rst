==============
Fusion physics
==============

Fusion physics in FAROES is as simple as possible. Only D-T fusion has been considered and only the simplest approximation for the rate coefficient has been implemented. 

The thermal D-T fusion rate coefficient is calculated (by :class:`.SimpleRateCoeff`) as

.. math::
   \left<\sigma v\right> = 1.1 \times 10^{-24}\,\mathrm{m}^3 \mathrm{s}^{-1}
                           (T_i/\mathrm{keV})^2,

which is a straight-line fit in log-log space to
the true rate coefficient curve at a temperature of around :math:`10\,\mathrm{keV}`.

This module also implements (in :class:`.NBIBeamTargetFusion`) a formula to account for beam-target fusion induced by NBI.

.. math::
        \mathrm{rate} = 80 * 1.1\cdot10^{14}\, \mathrm{s}^{-1}\,
        \frac{P_\mathrm{NBI}}{\mathrm{MW}}
        \left(\frac{\left<T_e\right>}{\mathrm{keV}}\right)^{3/2}

This formula is derived from work in :footcite:t:`strachan_fusion_1981`; the factor 80 represents the reactivity of D-T over that of the D-D experiments the fit was based on.


Implementation
==============

The components described here are used in various places in the zero-D tokamak plasma solver; they are not bundled together.

The reaction rate units are per-attosecond in order to give the drivers values near unity.

The components :class:`.SimpleRateCoeff`, :class:`.NBIBeamTargetFusion`, and :class:`.VolumetricThermalFusionRate` contain the 'physics'; :class:`.TotalDTFusionRate` is book-keeping and :class:`.SimpleFusionAlphaSource` is for convenience.

.. currentmodule:: faroes.fusionreaction

.. autoclass:: SimpleRateCoeff

.. autoclass:: NBIBeamTargetFusion

.. autoclass:: VolumetricThermalFusionRate

.. autoclass:: TotalDTFusionRate

:class:`.SimpleFusionAlphaSource` is the source of properties for calculations like the fast particle slowing time.

.. autoclass:: SimpleFusionAlphaSource

Future work
===========

The rate coefficient should be substituted for a more sophisticated model.
Reactions other than D-T should be considered, in particular the two D-D and D-3He.
The NBI beam-target fusion rates should be reviewed.
Separate tracking of the various fusion products and their energies.

Additional sophistication could involve components that account for (partial) polarization of the nuclear spins, which changes fusion rates and angles relative to the magnetic field at which the products are born. This leads to less isotropic neutron fluxes.

Another direction of increased sophistication would be treatment of fusion from various distribution functions, i.e. different parallel and perpendicular temperatures.

.. footbibliography::
