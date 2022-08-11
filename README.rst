******
FAROES
******

FAROES is "Fusion Analysis, Research, and Optimization for Energy Systems". It is a Python package for optimizing fusion power plants, and especially for optimizing properties like their capital cost or levelized cost of energy. It is built in a modular fashion to allow user-developers to modify or add equations or analyses of their own design.
It uses the OpenMDAO framework (openmdao.org) to provide interfaces to third-party nonlinear solvers and optimizers, especially gradient-based optimizers. The framework has advanced logging features to record solution data and metadata.
This framework also allows users to easily specify the design variables, constraints, and optimization targets, or to define their own.

|forthebadge made-with-python|

|pytest| |unittest|

.. |forthebadge made-with-python| image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/

.. |pytest| image:: https://github.com/cfe316/FAROES/workflows/pytests/badge.svg
   :target: https://github.com/cfe316/FAROES/workflows/pytests/badge
   :alt: Pytest build status

.. |unittest| image:: https://github.com/cfe316/FAROES/workflows/unittest-installs/badge.svg
   :target: https://github.com/cfe316/FAROES/workflows/unittest-installs/badge
   :alt: Unittest build status

Requirements
------------
* ``numpy>=1.21.0``
* ``scipy``
* ``openmdao == 3.15.0``. Can be installed with ``pip install "openmdao==3.15.0"``.
* ``ruamel >= 0.16``, a yaml parser.
* ``plasmapy`` (https://www.plasmapy.org/).

All these can be installed automatically using the ``setup.py`` file.


Installation
------------
Download the repo, ``cd`` to the folder and run ``pip install -e .``. This will download and install requirements and install FAROES in 'editable' mode.


Recommended Packages
---------------------
The two below can be installed on linux using the script available here: https://github.com/OpenMDAO/build_pyoptsparse/

* ``pyoptsparse`` (https://github.com/mdolab/pyoptsparse) in order to access the ``pyOptSparseDriver`` which allows use of more powerful optimizers like ``IPOPT``.
* ``IPOPT``. This is optimizer seems to perform a bit better than the implemntations of ``COBYLA`` or ``SLSQP`` which are included with ``scipy``.

Note that an environment variable must be set in every terminal instance using ``IPOPT``. We recommend adding a line to ``~/.bashrc`` to do this automatically for every terminal instance: ``export LD_LIBRARY_PATH=$HOME/ipopt/lib:$LD_LIBRARY_PATH``


Documentation Packages
--------------------
* ``docutils<0.17,>=0.15``
* ``nbclient<0.6,>=0.2``
* ``sphinx >= 4.4.0``
* ``jupyter-sphinx==0.3.2``
* ``Jinja2<3.1``
* ``sphinxcontrib-bibtex >= 2.3``
* ``jsonschema < 4``
* ``sphinx_rtd_theme``

The above packages are used to build the documentation for FAROES, and can be downloaded automatically by downloading the repos, ``cd`` to the folder, and running ``pip install -e .[docs]``, which will install FAROES in 'editable' mode alongside with the packages needed to build the documentation.
