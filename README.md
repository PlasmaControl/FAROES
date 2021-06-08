![Build Status](https://github.com/cfe316/FAROES/workflows/pytests/badge.svg)

# FAROES
Fusion Analysis, Research, and Optimization for Energy Systems

# Installation requirements
* `numpy`
* `scipy`
* `openmdao >= 3.8.0`. Can be installed with `pip install openmdao[all]`.
* `ruamel >= 0.16`, a yaml parser.
* [plasmapy](https://www.plasmapy.org/).

The latter three can be installed automatically using the `setup.py` file.

# Installation
Download the repo, `cd` to the folder and run `pip install -e .`. This will download and install requirements and install FAROES in 'editable' mode.

# Recommended packages
The two below can be installed on linux using the script available here: https://github.com/OpenMDAO/build_pyoptsparse/

* https://github.com/mdolab/pyoptsparse in order to access the `pyOptSparseDriver` which allows use of more powerful optimizers like `IPOPT`.
* IPOPT. This is optimizer seems to perform a bit better than the implemntations of `COBYLA` or `SLSQP` which are included with `scipy`.
