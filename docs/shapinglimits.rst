=====================================
Limits associated with plasma shaping
=====================================

Models for tokamaks
-------------------

Two models which describe the maximum practical tokamak elongation as a function of the aspect ratio have been implemented.

The first,

.. math:: κ = 0.95 (1.9 + 1.9 / (A^{1.4})),

is from :footcite:t:`menard_aspect_2004`, and the second,


.. math:: κ = 1.5 + 0.5 / (A - 1),

from :footcite:t:`zohm_physics_2013`.

These are implemented in :class:`.MenardKappaScaling` and :class:`.ZohmMaximumKappaScaling`.

Implementation details
----------------------

At present Menard's formula is used.

.. currentmodule:: faroes.shapinglimits

.. autoclass:: MenardKappaScaling

.. autoclass:: ZohmMaximumKappaScaling

Future work
-----------
It should be made possible to choose which formula is used in the model configuration.

.. footbibliography::

