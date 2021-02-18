Plasma Geometries
=================

The plasma geometry is often a starting point for a tokamak model. While most of the plasma physics itself is "zero-dimensional", this geometry specifies the space occupied by the plasma---the shape of the LCFS (last closed flux surface).

There are three plasma geometries available. The first and simplest is a pure ellipse,
the second is a modified ellipse-like shape which has a smaller cross section and volume, and the third is a parameterized shape described by Sauter which allows variable triangularity :math:`\delta` and squareness :math:`\xi`.

EllipseLikeGeometry
-------------------

The :code:`EllipseLikeGeometry` class is wrapped by the Groups :code:`EllipticalPlasmaGeometry` and :code:`MenardPlasmaGeometry`, below.
They bundle it with a :math:`\kappa` and/or :math:`\kappa_a` chosen based on the aspect ratio, from the :code:`MenardKappaScaling` class.

.. autoclass:: faroes.elliptical_plasma.EllipseLikeGeometry

.. autoclass:: faroes.elliptical_plasma.MenardKappaScaling

SauterGeometry
--------------

.. autoclass:: faroes.sauter_plasma.SauterGeometry

Convenience Groups
------------------
.. autoclass:: faroes.elliptical_plasma.EllipticalPlasmaGeometry

.. autoclass:: faroes.elliptical_plasma.MenardPlasmaGeometry

.. autoclass:: faroes.sauter_plasma.SauterPlasmaGeometry

.. autoclass:: faroes.sauter_plasma.SauterPlasmaGeometryMarginalKappa
