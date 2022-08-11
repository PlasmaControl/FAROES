=====================
Toroidal field ripple
=====================

A toroidal field ripple which is too large can cause a prompt loss of fast particles including fusion products.
This leads to less heating of the plasma and localized heating at the wall.
Fast particle loss calculations are expensive, so a constraint on the ripple magnitude is used as a proxy.
The treatment is from :footcite:t:`wesson_tokamaks_2004`.

.. math::
        \delta = \left(\frac{R}{r_2}\right)^{N} + \left(\frac{r_1}{R}\right)^{N}

where :math:`N` is the number of coils,
:math:`r_1` and :math:`r_2` are the average conductor radii of the inboard and outboard legs, respectively,
and :math:`R` is the major radius of the location to be evaluated.
Generally, constraints are applied to limit the ripple at the inboard and outboard midplane and at the geometric plasma center.

.. currentmodule:: faroes.ripple

.. autoclass:: SimpleRipple

.. footbibliography::
