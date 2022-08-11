=================
Trapped particles
=================

Estimates for the fraction of trapped particles on a flux surface are used in the calculation of the :doc:`bootstrap` and in the neutral beam current drive calculation.


Models for Zero-D Tokamaks
--------------------------

There are two implemented models for estimates of the trapped particle fraction on a flux surface.
The first, :class:`.TrappedParticleFractionUpperEst``, is from :footcite:t:`linliu_upper_1995`, and relies only on the flux surface's inverse aspect ratio :math:`\epsilon`. The second, :class:`.SauterTrappedParticleFraction`, is from :footcite:t:`sauter_geometric_2016` and also takes into account the flux surface's triangularity :math:`\delta`.

Both of these are calculations for a particular flux surface. It should be a 'typical' flux surface for the phenomenon at hand.

.. currentmodule:: faroes.trappedparticles

.. autoclass:: TrappedParticleFractionUpperEst

.. autoclass:: SauterTrappedParticleFraction

.. footbibliography::
