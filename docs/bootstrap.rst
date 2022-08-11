=================
Bootstrap current
=================

Models for Zero-D Tokamaks
--------------------------

The :class:`.BootstrapCurrent` group calculates the necessary bootstrap current fraction to yield a plasma with a specified total current, given the inverse aspect ratio, poloidal beta, cylindrical safety factor, and optionally, the triangularity. This is by necessity a highly simplified zero-D calculation, and does not reference any notion of profiles.

The group uses the components :class:`.BootstrapFraction`, :class:`.BootstrapMultiplier`, and :class:`.SauterBootstrapProportionality`. The last is an ad-hoc adjustment for the triangularity.

.. currentmodule:: faroes.bootstrap

.. autoclass:: BootstrapCurrent

.. autoclass:: BootstrapFraction

.. autoclass:: BootstrapMultiplier

.. autoclass:: SauterBootstrapProportionality

.. footbibliography::
