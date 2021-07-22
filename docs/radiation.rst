Radiation
=========

Radiation is an important factor of consideration for tokamaks. There are three main modes of radiation in tokamaks; Bremsstrahlung, synchrotron, and impurity radiation. 

Bremsstrahlung
--------------
The :code:`Bremsstrahlung` class draws from the other classes below that are dependent on profile and triangularity. The Bremsstrahlung formluation varies depending on whether the temperature and density profiles are constant, parabolic, or pedestal-shaped, and whether the triangularity is a constant or linear function of :math:`\rho`.

.. autoclass:: faroes.shapefactor.ConstProfile
.. autoclass:: faroes.shapefactor.ParabProfileConstTriang
.. autoclass:: faroes.shapefactor.ParabProfileLinearTriang

.. autoclass:: faroes.bremsstrahlung.PedestalProfileConstTriang
.. autoclass:: faroes.bremsstrahlung.PedestalProfileLinearTriang
.. autoclass:: faroes.bremsstrahlung.Bremsstrahlung


Synchrotron
-----------

.. autoclass:: faroes.synchrotron.SynchrotronFit
.. autoclass:: faroes.synchrotron.Synchrotron


Impurity
--------

