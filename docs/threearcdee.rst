=========================
Three-arc Dee coil shapes
=========================

The two 'three-arc-Dee' shapes are simple paramaterizations for the shape of the toroidal field coils.
The first, :class:`.ThreeArcDeeTFSet`, is a special case of the second, :class:`.ThreeEllipseArcDeeTFSet`.

There are also two "Adapter" classes, :class:`.ThreeArcDeeTFSetAdaptor` and :class:`.ThreeEllipseArcDeeTFSetAdaptor`. Using these decreases the space of *infeasible* designs which are explored.
They are particularly useful when the magnets must fit around the blanket/neutron shield, which already have an established height and outer radius.

.. currentmodule:: faroes.threearcdeecoil

.. autoclass:: ThreeArcDeeTFSet

.. autoclass:: ThreeEllipseArcDeeTFSet

.. autoclass:: ThreeArcDeeTFSetAdaptor

.. autoclass:: ThreeEllipseArcDeeTFSetAdaptor

See also
--------
- :doc:`princetondee`
