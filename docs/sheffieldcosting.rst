======================
Sheffield costing code
======================

The Sheffield tokamak costing code, first published in 1986 :footcite:p:`sheffield_cost_1986`, is largely based on the STARFIRE costing effort from 1980 :footcite:p:`baker_starfire_1980-1`. The Sheffield model was updated in 2016 :footcite:p:`sheffield_generic_2016` to incorporate estimated costs of ITER.

This module implements the Sheffield model as closely as possible.
Alternately, there is an option to implement some slight improvements via setting :code:`exact_generomak=False` as an option of :class:`.GeneromakCosting`, the top-level group; see the Notes of that function for details.

The costing code and the definitions of the various components are somewhat subtle. See the papers for the reference descriptions of what each component represents.

Implementation
==============

:class:`.GeneromakCosting` is the top-level group.

.. currentmodule:: faroes.generomakcosting

.. autoclass:: GeneromakCosting

Cost coefficients
-----------------

There are roughly 35 coefficients associated with the costs of various components and the size of the reference plant from which the values are scaled.
These are found in the configuration tree; shown are the values from the 2016 paper::

 costing:
   Generomak:
     Finance:
       F_CR0: 0.078
     PrimaryCoils:
       cost per volume: {value: 1.66, units: MUSD/m**3}
     Blanket:
       cost per volume: {value: 0.75, units: MUSD/m**3}
       f_failures: 1.1
       f_spares: 1.1
       fudge: 1.0
     Divertor:
       cost per area: {value: 0.114, units: MUSD/m**2}
       f_failures: 1.2
       f_spares: 1.1
       fudge: 1.0
     Structure:
       cost per volume: {value: 0.36, units: MUSD/m**3}
     ShieldWithGaps:
       cost per volume: {value: 0.29, units: MUSD/m**3}
     AuxHeating:
       cost per watt: {value: 5.3, units: MUSD/MW}
       f_spares: 1.1
       annual maintenance factor: 0.1
     ReferencePlant:
       d_Pt: {value: 4.150, units: GW}
       e_Pt: 0.6
       d_Pe: {value: 1.200, units: GW}
     FusionIsland:
       c_Pt: {value: 0.221, units: GUSD}
       m_pc: 1.5
       m_sg: 1.25
       m_st: 1.0
       fudge: 1.0
     CapitalCost:
       contingency: 1.15
       c_e1: {value: 0.900, units: GUSD}
       c_e2: {value: 0.900, units: GUSD}
       c_V: {value: 0.839, units: GUSD}
       d_V: {value: 5100, units: m**3}
       e_V: 0.67
       fudge: 1.0  # not implemented yet
     MiscReplacements:
       c_misc: {value: 52.8, units: MUSD/a}
     FixedOM:
       base_cost: {value: 108, units: MUSD/a}
       fudge: 1.0
     WasteCharge: {value: 0.5, units: mUSD/kW/h}
     Consumables:
       Deuterium:
         cost_per_mass: {value: 10, units: kUSD/kg}
     COE:
       fudge: 1.0

These values are loaded by :class:`.GeneromakCosting` and 'repackaged' into smaller dicts in order to be sent to the various subcomponents as options. This is done to allow a switch to a different configuration method in the future.

Estimated plant dimensions
--------------------------
.. autoclass:: GeneromakStructureVolume

Costs of parts of the plant
---------------------------

.. autoclass:: PrimaryCoilSetCost

.. autoclass:: BlanketCost

.. autoclass:: StructureCost

.. autoclass:: ShieldWithGapsCost

.. autoclass:: DivertorCost

.. autoclass:: AuxHeatingCost

.. autoclass:: FusionIslandCost

.. autoclass:: DeuteriumVariableCost


Annual costs
------------

.. autoclass:: AnnualAuxHeatingCost

.. autoclass:: MiscReplacements

.. autoclass:: AveragedAnnualBlanketCost

.. autoclass:: AveragedAnnualDivertorCost

.. autoclass:: AnnualDeuteriumCost

.. autoclass:: FuelCycleCost

.. autoclass:: FixedOMCost

Total costs
-----------

.. autoclass:: CapitalCost

Construction and Financing
--------------------------

.. autoclass:: TotalCapitalCost

.. autoclass:: IndirectChargesFactor

.. autoclass:: ConstantDollarCapitalizationFactor

Cost of electricity
-------------------

.. autoclass:: CostOfElectricity

.. footbibliography::

See also
========

- :doc:`genxinterface`
