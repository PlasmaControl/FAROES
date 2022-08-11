===========================
Fast particle slowing times
===========================

When alpha particles are born or neutral beam particles are ionized in the plasma, they are much faster than the surrounding near-thermal ions. They can take a relatively long time to slow down, typically milliseconds to seconds. While slowing down, they typically heat the thermal *electrons* more than they do the thermal ions. They also represent a significant fraction of the total plasma pressure. Too high a fraction of the pressure coming from fast particles can pose a problem as there are various fast particle-related instabilities.

Calculation method, theory
==========================
A prerequisite is the velocity slowing time of fast ions on electrons, often also called the Spitzer slowing down time, :math:`\tau_\mathrm{se}`.
This is valid in the regime where the fast ions are moving slower than the electron thermal velocity.
It is defined such that

    .. math:: du/dt = - u / \tau_s

and is calculated as

    .. math::

       \tau_\mathrm{s,e} = c \frac{A_b (T_e / \mathrm{keV})^{3/2}}
           {Z^2 (n_e / (10^{20} \mathrm{m}^{-3}) \log{\Lambda_e}}

where

    .. math::

       c &= \frac{3 \pi^{3/2} \epsilon_0^2 m_e \mathrm{u}}
          {10^{20}\,\mathrm{m}^{-3} e^4}
          \left(\frac{2\,\mathrm{keV}}{m_e}\right)^{3/2}.

         &= 0.19834312\;\mathrm{s}

This computation is handled by :class:`.SlowingTimeOnElectrons`.

The second computation is the ion's critical slowing energy, :math:`W_\mathrm{crit}` or :math:`W_c`.
At high speeds (above :math:`W_c`), most of the energy removed from the fast ion as it slows down goes to heating the electrons,
while below :math:`W_c` most of the energy goes to heating the thermal ions.

There are two methods (components) for calculating :math:`W_c`, one following :footcite:t:`stix_heating_1972` (:class:`.StixCriticalSlowingEnergy`) and one following :footcite:t:`bellan_fundamentals_2006` ( :class:`.BellanCriticalSlowingEnergy`):

    .. math::

       (\textrm{Stix})\quad \alpha' &= A_t \sum_i n_i Z_i^2 / A_i

       (\textrm{Bellan})\quad \alpha' &= \sum_i n_i Z_i^2 (1 + A_t / A_i)

       \beta' &= \frac{4}{3 \pi^{1/2}} n_e

       W_\mathrm{crit} &= T_e \left(\frac{m_T}{m_e}\right)^{1/3}
           \left(\frac{\alpha'}{\beta'}\right)^{2/3}

They are identical other than the calculation of :math:`\alpha'`. The sum indexes over the background plasma ions. The subscript :math:`_T` refers to the fast ion as the test particle.

Next we can start to calculate properties of the ever-slowing population of fast particles.

Particles slow down more and more quickly as they lose energy, and in the model they will lose all their energy in a finite time. (Of course in reality they will simply become thermalized.) This time is

    .. math::

       \tau_\mathrm{th} = \frac{\tau_\mathrm{se}}{3}
           \log\left(1 +
           \left(\frac{W}{W_\mathrm{crit}}\right)^{3/2}\right),

and is implemented by :class:`.SlowingThermalizationTime`.

As they slow, the fast ions give some of their energy to the background electrons and some to ions. :footcite:t:`stix_heating_1972` computes the fractions which go to ions and electrons as

    .. math::

       f_i &= \frac{W_c}{W} \int_0^{W/W_c} \frac{dy}{1 + y^{3/2}}

         &= \, _2F_1\left(\frac{2}{3},1;\frac{5}{3};-(W/W_c)^{3/2}\right)

       f_e &= 1 - f_i;

this is implemented by :class:`.FastParticleHeatingFractions`.

The thermalization processes is nonlinear and during the thermalization time :math:`\tau_\mathrm{th}` the particles have some average energy,

    .. math::

        \bar{W} = \frac{1}{\tau_\mathrm{th}} \int_0^{\tau_{th}} W(t) \; dt

This integral evaluates to

    .. math::

        \bar{W} = \frac{W_c}{6 \log(1 + W_r)} \left(-4\cdot\, 3^{1/2} \pi +
          9(1 + W_r^{3/2})^{2/3}
             \, _2F_1\left(-\frac{2}{3}, -\frac{2}{3}; \frac{1}{3};
             \frac{1}{1 + W_r^{3/2}} \right)\right).

where :math:`W_r \equiv W/W_c`. This function is implemented by :class:`.AverageEnergyWhileSlowing`.

Finally, to find the total energy of the fast particles of a particular type in the plasma, we need to know their source rate :math:`S`. Their total energy is then

  .. math:: W_\mathrm{fast} = \bar{W}\,\tau_\mathrm{th}\,S.

This final calculation is implemented in :class:`.FastParticleSlowing`.

Models for Zero-D tokamaks
==========================

In order to compensate for having to use the averaged values :math:`\left<n_i\right>` and :math:`\left<T_e\right>` we make an adjustment to :math:`W_\mathrm{crit}` of the form

    .. math::

        W_\mathrm{crit} = \mathrm{scale} \; W_{\mathrm{crit},0}.

Implementation
==============

Top-level group
---------------

The top-level group is :class:`.FastParticleSlowing`.

.. currentmodule:: faroes.fastparticleslowing

.. autoclass:: FastParticleSlowing

Spitzer slowing down time
-------------------------

.. autoclass:: SlowingTimeOnElectrons

Critical slowing energy models
------------------------------

.. autoclass:: StixCriticalSlowingEnergy

.. autoclass:: BellanCriticalSlowingEnergy

.. autoclass:: CriticalSlowingEnergyRatio

Outcomes of the slowing-down distribution
-----------------------------------------

.. autoclass:: SlowingThermalizationTime

.. autoclass:: FastParticleHeatingFractions

.. autoclass:: AverageEnergyWhileSlowing

Usage hints, further projects
-----------------------------
In the zero-D tokamak model, :class:`.FastParticlesSlowing` is used twice; once for the alphas and once for a neutral beam ion. It is not yet 'vectorized' to handle multiple initial energies of fast particles, but probably could be: this would be useful for handling multiple fusion products or multiple NBI ion energy fractions.

.. footbibliography::
