############
Installation
############

Requirements
------------
* ``numpy``
* ``scipy``
* ``openmdao == 3.15.0``.
* ``ruamel >= 0.16``, a yaml parser.
* ``plasmapy`` (https://www.plasmapy.org/)

All of these can be installed automatically using the ``setup.py`` file.

Installation
------------
Download the repo, ``cd`` to the folder and run ``pip install -e .``. This will download and install requirements and install FAROES in 'editable' mode.

Recommended packages
---------------------
The two below can be installed on linux using the script available here: https://github.com/OpenMDAO/build_pyoptsparse/

* ``pyoptsparse`` (https://github.com/mdolab/pyoptsparse) in order to access the ``pyOptSparseDriver`` which allows use of more powerful optimizers like ``IPOPT``.
* ``IPOPT``. This is optimizer seems to perform a bit better than the implemntations of ``COBYLA`` or ``SLSQP`` which are included with ``scipy``.
