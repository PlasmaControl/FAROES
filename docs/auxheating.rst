==================
Auxilliary heating
==================

Two methods for plasma heating have been implemented: neutral beam injection (NBI), which heats and drives current, and heating-only RF.

Models for zero-D tokamaks
==========================

Radiofrequency heating
----------------------

Radiofrequency heating is simply extra heating power for the plasma. It does not drive current. It requires electrical energy :math:`P_\mathrm{aux}` which is larger than the heating power :math:`P` by an efficiency factor:

.. math:: P_\mathrm{aux} = P / \mathrm{eff}.

It may be more energy-efficient or cost-efficient than the heating power provided by neutral beams.

The power and efficiency are used by the :class:`.AuxilliaryPower` group.
The cost scales with the power in the :doc:`sheffieldcosting` model.

RF can also be used to drive current but this is not implemented.


Neutral beam injection heating and current drive
-------------------------------------------------

The neutral beam heating system both heats the plasma and drives current.
It injects ions of a certain species and ionization level with some injected power, for example, 14 MW of 500 keV deuterium ions. The energy and injected power are common design variables.

The injection energy changes the efficiency of the current drive; see :doc:`nbi_current_drive`.

The total source rate of the ions is

.. math:: S = E / P.

One the current drive efficiency, :math:`I_t/P` is calculated, the total beam-driven current can be computed by :class:`.NBICurrent`,

.. math::
        I_\mathrm{NBI} = \mathrm{fudge} \,S\, E \frac{It}{P}.

In reality, NBI systems also inject particles (at rate :math:`S`) and momentum into the plasma. These are not treated.

NBI heating systems are subject to significant limitations in real power plants. If the plasma density is too low there can be 'shine-through' where neutrals pass through the plasma without ionizing.
The the density is too high, the ions may be deposited near the plasma edge, where the fast ions may escape. Neutral beams are also physically very large and take up significant port space. These aspects of the NBI system are not treated.

As part of the power plant
^^^^^^^^^^^^^^^^^^^^^^^^^^

The NBI systems have some wall-plug efficiency which relates the power by which the plasma is heated and the electrical power required to drive the system. This takes the form of a specified efficiency,

.. math:: P_\mathrm{aux} = P / \mathrm{eff}.

The power and efficiency are used by the :class:`.AuxilliaryPower` group.
The cost scales with the power in the :doc:`sheffieldcosting` model.

Total auxilliary power
----------------------
The powers of the RF and NBI systems are summed together to find the total auxilliary heating power and wall-plug electric power.

    .. math::
       P_\mathrm{aux,h} &= P_\mathrm{NBI} + P_\mathrm{RF}

       P_\mathrm{aux,e} &= P_\mathrm{NBI}/η_\mathrm{NBI} +
                          P_\mathrm{RF}/η_\mathrm{RF}

Implementation
==============

RF heating
----------

.. autoclass:: faroes.rfheating.SimpleRFHeating

.. autoclass:: faroes.rfheating.SimpleRFHeatingProperties

NBI heating
-----------

.. autoclass:: faroes.nbisource.SimpleNBISource

.. autoclass:: faroes.nbisource.SimpleNBISourceProperties

The :class:`.NBICurrent` component is designed to handle multiple species of injected ions which each have their own current drive efficiency; this is not yet used in the zero-D tokamak models.

.. autoclass:: faroes.nbicd.NBICurrent

Total Auxilliary Power
----------------------

.. autoclass:: faroes.powerplant.AuxilliaryPower

See also
--------
- :doc:`nbi_current_drive`
- :doc:`powerplant`
- :doc:`selfconsistentplasma`
- :doc:`sheffieldcosting`

.. footbibliography::
