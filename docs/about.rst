About FAROES
============

FAROES stands for Fusion Analysis, Research, and Optimization for Energy Systems.

It is an open-source 'fusion systems code', with the initial goal of allowing rough costing studies for tokamak fusion reactors. It currently offers a 'zero-dimenional' steady-state tokamak model. Costing is done using the Sheffield formulation, where cost scales with the volume of various major components and with the thermal and electric power generated.

The package is built using the `OpenMDAO <https://openmdao.org/>`_ (Multidisciplinary Design, Analysis, and Optimization) framework. This allows access to nonlinear solvers and gradient-based optimizers and manages the model's Jacobian. Running the code with non-gradient-based optimizers is also possible.

This code is developed using funding from the Department of Energy, including Contract No. DE-AC02-09CH11466.

Why FAROES?
-----------
The code is named for the Faroe Islands, a archipelago in the North Atlantic and part of the Kingdom of Denmark.
The islands have a rugged and elemental appearance. The hills are not obscured by trees, so you can see every form and detail, which is reflected in the code's open-source nature.
The glaciers carved valleys and fjords with wide, uniform profiles, which gives a 'geometric' appearance to the terrain. This is reflected in the code's zero-dimensional nature.

.. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Faroe_Islands%2C_Bor%C3%B0oy%2C_Klaksv%C3%ADk_%283%29.jpg/1024px-Faroe_Islands%2C_Bor%C3%B0oy%2C_Klaksv%C3%ADk_%283%29.jpg
   :alt: A photograph of a fjord with a town on each side.
   :align: center

   Klaksvík, on the island of Borðoy, is the Faroe Islands' second-largest town.
   `Vincent van Zeijst <https://commons.wikimedia.org/wiki/File:Faroe_Islands,_Bor%C3%B0oy,_Klaksv%C3%ADk_(3).jpg>`_, `CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>`_, via Wikimedia Commons.

   ..

History
-------
FAROES started as a port of a 600-line spreadsheet model developed at PPPL.
