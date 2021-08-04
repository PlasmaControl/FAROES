Bremsstrahlung
==============
The :code:`Bremsstrahlung` class draws from the other classes below that are dependent on profile and triangularity. The Bremsstrahlung formluation varies depending on whether the temperature and density profiles are constant, parabolic, or pedestal-shaped, and whether the triangularity is a constant or linear function of :math:`\rho`.

.. autoclass:: faroes.bremsstrahlung.Bremsstrahlung

Profiles
--------
.. autoclass:: faroes.shapefactor.ConstProfile

.. autoclass:: faroes.shapefactor.ParabProfileConstTriang

.. autoclass:: faroes.shapefactor.ParabProfileLinearTriang

.. autoclass:: faroes.bremsstrahlung.PedestalProfileConstTriang

.. autoclass:: faroes.bremsstrahlung.PedestalProfileLinearTriang


