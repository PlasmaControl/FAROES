=================
Plasma Geometries
=================

The plasma geometry is often a starting point for a tokamak model. While most of the plasma physics itself is "zero-dimensional", this geometry specifies the space occupied by the plasma---the shape of the LCFS (last closed flux surface).

There are three plasma geometries available. The first and simplest is a pure ellipse,
the second is a modified ellipse-like shape which has a smaller cross section and volume, and the third is a parameterized shape described by Sauter which allows variable triangularity :math:`δ` and squareness :math:`ξ`.

Implementation
--------------
The Sauter geometry implementation is also more sophisticated; it outputs arrays which describe the boundary shape. This allows the first wall, blanket, shield, and magnets to conform to the shape of the plasma.

EllipseLikeGeometry
-------------------

The :class:`.EllipseLikeGeometry` class is wrapped by the Groups :class:`.EllipticalPlasmaGeometry` and :class:`.MenardPlasmaGeometry`, below.
They bundle it with a :math:`κ` and/or :math:`κ_a` chosen based on the aspect ratio, from the :class:`.MenardKappaScaling` class.

.. autoclass:: faroes.elliptical_plasma.EllipseLikeGeometry

SauterGeometry
--------------

.. autoclass:: faroes.sauter_plasma.SauterGeometry

Convenience Groups
------------------
.. autoclass:: faroes.elliptical_plasma.EllipticalPlasmaGeometry

.. autoclass:: faroes.elliptical_plasma.MenardPlasmaGeometry

.. autoclass:: faroes.sauter_plasma.SauterPlasmaGeometryMarginalKappa

.. footbibliography::
