########################
Energy confinement times
########################

Models for Zero-D Tokamaks
--------------------------

It's difficult to predict the energy confinement time :math:`\tau_e` in tokamaks from first principles. Instead, tokamak researchers typically use scaling laws cobbled together from experimental datasets. These take the form of power law fits,

.. math:: \tau_{e} = c_0 \sum_i A_i^{c_i},

where the :math:`A_i` and :math:`c_i` are lists of interesting, and hopefully easily-measured quantities, and exponents derived from a power law fit, respectively. Typical quantities used for the :math:`A_i` include :math:`I_p`, the plasma current; :math:`B_t`, the vacuum toroidal field at the geometric plasma center; :math:`n_e`, a typical electron density for the plasma; :math:`P_L`, either the plasma heating power or the power lost by conduction and turbulence out the edge of the plasma; :math:`R`, the major radius of the plasma geometric center; :math:`\epsilon`, the inverse aspect ratio; :math:`\kappa_a`, the effective elongation; and :math:`M`, the main ion mass number (i.e. 2 if pure deuterium and 2.5 for DT).

Of course, effects such as the plasma shaping, special profiles, and other plasma physics mechanisms alter the energy confinement time so that it does exactly match the power law fit.
The ratio of the actual energy confinement time to that expected is conventially denoted :math:`H`, and called the "H-factor".

.. math:: \tau_e = H \tau_{e, \mathrm{law}}

The :class:`.ConfinementTime` group computes :math:`\tau_e` from the various inputs :math:`A_i` and :math:`H`.
It uses the :class:`.ConfinementTimeScaling` component to generate the expected :math:`\tau_e`, and then :class:`.ConfinementTimeMultiplication` scales it by :math:`H`.

Many confinement time scaling laws have been derived over the years using various datasets.
The :class:`.ConfinementTime` group has a string option ``scaling`` to choose the law. The choices are

+--------------+---------------------------------------+
| scaling      | Reference                             |
+==============+=======================================+
| H98y2        | :footcite:t:`doyle_chapter_2007`      |
+--------------+---------------------------------------+
| H89P         | :footcite:t:`yushmanov_scalings_1990` |
+--------------+---------------------------------------+
| Petty        | :footcite:t:`petty_sizing_2008`       |
+--------------+---------------------------------------+
| MenardHybrid | see :class:`.MenardHybridScaling`     |
+--------------+---------------------------------------+
| NSTX-MG      |                                       |
+--------------+---------------------------------------+
| MAST-MG      |                                       |
+--------------+---------------------------------------+
| NSTX-design  |                                       |
+--------------+---------------------------------------+
| User         | see below                             |
+--------------+---------------------------------------+

and additionally the choices ``"default"`` and ``None`` lead to "H98y2".

The :math:`A_i, c_i` data associated with each law are loaded from ``fits.yaml``.
See that file for the constants associated with each law.
The "User" scaling law allows a definition of a user-defined law using a dict.
The input variables must be a subset of those listed above.

The MenardHybrid law is a special choice which interpolates between NSTX-MG scaling at small aspect ratio and Petty scaling at large aspect ratio. See :class:`.MenardHybridScaling`.

.. currentmodule:: faroes.confinementtime

.. autoclass:: ConfinementTime

.. autoclass:: ConfinementTimeScaling

.. autoclass:: ConfinementTimeMultiplication

Additional scaling laws
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MenardHybridScaling

.. footbibliography::
