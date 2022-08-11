=====================================
Neutral beam current drive efficiency
=====================================

Models for Zero-D Tokamaks
--------------------------

The model for the neutral beam current drive efficiency is separate from the actual calculation of the driven current.
This calculation of the 'efficiency', in terms of amperes of toroidal plasma current driven per watt of injected power, is more involved.

The model here assumes that the beam is injected into a uniform toroidal plasma.
The calculation generally follows the treatment of :footcite:t:`start_effect_1980` Equation (45),
but is slightly altered and simplfied.
The new calculation results in

    .. math::

        \frac{I_t}{P} =& \frac{\tau_{se} v_0 Z_b G}
            {2 \pi R (1 + \alpha^3) E_{NBI}}

        & \left(1 + (3 - 2 \alpha^3 \beta_1) \delta / (1 + \alpha^3)^2\right)

        & \int_0^1 x^{3 + \beta_1}
           \left(\frac{1 + \alpha^3}{x^3 + \alpha^3}\right)^{1+\beta_1/3} \; dx

where :math:`\delta \equiv \left<T_e\right>/(2 E_\mathrm{NBI})` and :math:`\tau_{se}` is the Spitzer velocity slowing time of the beam ions on the background electrons. See :class:`.CurrentDriveAlphaCubed` and :class:`.CurrentDriveBeta1` for definitions of :math:`\alpha^3` and :math:`\beta_1`.


The simplification is that it assumes injection in a purely parallel direction, which is equivalent to setting :math:`K_1 = 1` in Start.
It also does not assume a large aspect ratio tokamak. This leads to an alteration of the term
:math:`(1 - Z_b/Z_\mathrm{eff} + 1.46 \epsilon^{1/2} A Z_b/Z_\mathrm{eff})` in Equation (45) to a term
:math:`G` which is constructed using :class:`.CurrentDriveG` and :class:`.CurrentDriveA`,

.. math::

   G = 1 + \left(f_\mathrm{trap,u} \, \left(1
       + \frac{0.6}{(1 + v_B/v_{\mathrm{th},e})Z_\mathrm{eff}}\right) - 1 \right)
         \frac{Z_b}{Z_\mathrm{eff}}.

It also leaves out Start's term :math:`\gamma`, which is not defined in the paper.

In general, this model does not consider neutral beam shine-through nor deposition profiles.

The main group to call is :class:`.CurrentDriveEfficiency`.

.. currentmodule:: faroes.nbicd

.. autoclass:: CurrentDriveEfficiency

Calculation details
^^^^^^^^^^^^^^^^^^^
Several components perform the top group's calculations.

.. autoclass:: CurrentDriveProperties

.. autoclass:: CurrentDriveBeta1

.. autoclass:: CurrentDriveA

.. autoclass:: CurrentDriveAlphaCubed

.. autoclass:: CurrentDriveG

.. autoclass:: CurrentDriveEfficiencyTerms

.. autoclass:: CurrentDriveEfficiencyEquation

.. footbibliography::
