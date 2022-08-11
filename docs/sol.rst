================
SOL and Divertor
================

Many tokamak designs are limited by the extreme heat fluxes at the divertor.

In systems modeling, the peak heat flux on the divertor is often used as a constraint. In the Generomak costing model, the peak heat flux affects the divertor's operational life before replacement, and therefore the availability of the overall reactor core.

Calculation
===========

There are four steps in calculating the divertor heat flux.

#. Determine the strike point radius
#. Determine power to the SOL
#. Computing the upstream SOL width,
#. Calculate the peak divertor heat flux based on the strike point geometry.

The first two steps are independent of each other. The first could be considered part of the overall plasma geometry. Steps 3 and 4 occur in sequence after the first two.

1. strike point radius
-----------------------
There are currently three models for the strike point radius, the snowflake divertor (SF), the Super-X divertor (SXD), and the LinearDelta model.

This is computed in one of three ways. For the snowflake divertor (SF),

.. math::

   R_\mathrm{strike} = R_0 - c a

where :math:`c` is a constant and :math:`a` is the minor radius.

For the super-X divertor (SXD),

.. math::

    R_\mathrm{strike} = c R_0

where c is a (different) constant. And for the LinearDelta model,

.. math::

    R_\mathrm{strike} = R_0 + f_w a - f_d \delta a,

where :math:`f_w` and :math:`f_d` are two constants and :math:`\delta` is the triangularity of the LCFS.

These models are implemented in :class:`.StrikePointRadius`.

2. power to the SOL
--------------------

The power to the SOL is

.. math:: P_\mathrm{SOL} = P_\mathrm{heat} (1 - f_\mathrm{rad}).

where :math:`P_\mathrm{heat}` is the total (external and alpha) heating power of the core and :math:`f_\mathrm{rad}` is the fraction of that power which is radiated in the core---the remainder spills out into the SOL.

In the tokamak model, this :math:`f_\mathrm{rad}` is not connected to the :math:`f_\mathrm{rad}` specified or computed for the core plasma physics.

This is implemented via an ExecComp in :class:`.SOLAndDivertor`.

3. width of the scrape-off layer
--------------------------------
Currently there is only one model.

The Goldston HD model :footcite:p:`goldston_heuristic_2012` is used to calculate the SOL heat flux width :math:`\lambda_\mathrm{HD}`.

    .. math::

       \lambda_\mathrm{HD}/\mathrm{mm} =
           5.761 \left(\frac{P_\mathrm{SOL}}{\mathrm{W}}\right)^{1/8}
           (1 + \kappa^2)^{5/8}
           \left(\frac{a}{\mathrm{m}}\right)^{17/8}
           \left(\frac{B_T }{ \mathrm{T}}\right)^{1/4} \\
           \left(\frac{I_p }{ \mathrm{A}}\right)^{-9/8}
           \left(\frac{R}{\mathrm{m}}\right)^{-1}
           \left(\frac{2 \bar{A}}{\bar{Z}^2(1 + \bar{Z})}\right)^{7/16}
           \left(\frac{Z_\mathrm{eff} + 4}{5}\right)^{1/8}

The SOL plasma composition (:math:`\bar{A}`, :math:`\bar{Z}`, and :math:`Z_\mathrm{eff}`) is specified separately from the composition of the core plasma.

The Goldston model also predicts the upstream SOL temperature. In principle this could be used as a constraint or check on a core-edge model.

    .. math::

       T_\mathrm{sep}/\mathrm{eV} =
           30.81 \left(\frac{P_\mathrm{SOL}}{\mathrm{W}}\right)^{1/4}
           \left(\frac{1 + \bar{Z}}{2 \bar{A}}\right)^{1/8}
           \left(\frac{a}{\mathrm{m}}\right)^{1/4} \\
           \left(1+\kappa^2\right)^{1/4}
           \left(\frac{B_T}{\mathrm{T}}\right)^{1/2}
           \left(\frac{I_p}{\mathrm{A}}\right)^{-1/4}
           \left(\frac{Z_\mathrm{eff} + 4}{5}\right)^{1/4}

This model is implemented by :class:`.GoldstonHDSOL`.

When using :class:`.SOLAndDivertor`, the SOL heat flux width can be further adjusted via a fudge factor (loaded by :class:`.SOLProperties`) to match a more sophisticated calculation:

.. math::

   \lambda_\mathrm{SOL} = \mathrm{fudge}\; \lambda_\mathrm{HD}.


4. peak heat flux at the target
-------------------------------

There are two models for calculating the peak target heat flux, the poloidal angle model and the total angle model.

In the poloidal angle model,

.. math::

   q_{max} = P_\mathrm{SOL} \frac{f_\mathrm{outer}}
            {N_\mathrm{div} 2 \pi R_\mathrm{strike}
            \lambda_\mathrm{SOL}} \frac{\sin(\theta_\mathrm{pol})}{
            f_\mathrm{flux exp}}.

Here :math:`f_\mathrm{outer}` is the fraction of the SOL power which goes to the outer divertor. (The outer divertor generally has a more critical heat flux than the inner.)
:math:`N_\mathrm{div}` is the number of divertors, typically either 1 or 2.
:math:`f_\mathrm{fluxexp}` is the factor by which the poloidal field has expanded at the strike point compared with the outer midplane location where :math:`\lambda_\mathrm{SOL}` is defined.
:math:`\theta_\mathrm{pol}` is the poloidal angle at which the field lines intersect the divertor surface at the strike point.

In the total angle model,

.. math::

   q_{max} = P_\mathrm{SOL} \frac{f_\mathrm{outer}}
            {N_\mathrm{div} 2 \pi R_\mathrm{strike}
            \lambda_\mathrm{SOL}} \frac{q^*}{\kappa}
                    \sin(\theta_\mathrm{tot}).

Here :math:`q^*` is the cylindrical safety factor and :math:`\theta_\mathrm{tot}` is the total angle at which the field line approaches the target. (This is likely to be quite shallow as the field lines are mostly toroidal.)

These are implemented by :class:`.PeakHeatFlux`.

Implementation
==============

The :class:`.SOLAndDivertor` group incorporates the :class:`.GoldstonHDSOL`, :class:`.StrikePointRadius`, and :class:`PeakHeatFlux` components, and the SOL-specific properties loader :class:`.SOLProperties`.

.. currentmodule:: faroes.sol

.. autoclass:: SOLAndDivertor

Properties
----------

The properties loaded from the configuration tree by this class are not related or connected to their values in the core plasma.

.. autoclass:: SOLProperties

Strike point radius
-------------------

.. autoclass:: StrikePointRadius

Goldston HD model
-----------------

.. autoclass:: GoldstonHDSOL

Strike point geometry
---------------------

There are two methods of representing the geometry of the strike point.

.. autoclass:: PeakHeatFlux


.. footbibliography::

