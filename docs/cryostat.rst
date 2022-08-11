=============
Cryostat size
=============

The dimensions of the machine cryostat are often used in cost scaling models.

Models for zero-D tokamaks
--------------------------

The :class:`.SimpleCryostat` describes a cryostat which surrounds all the TF coils, similar to in ITER. It is a vacuum vessel, but not the primary vacuum boundary, as the inner vacuum vessel is nested inside within the TFs.

.. currentmodule:: faroes.cryostat

.. autoclass:: SimpleCryostat

See also
--------
- :doc:`sheffieldcosting`
