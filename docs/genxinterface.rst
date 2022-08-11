----------------------------------------
Interface from Sheffield costing to GenX
----------------------------------------

This is an interface from the :doc:`sheffieldcosting`
to the cost-related inputs required by a standard
`GenX`_ generator, especially for the file `generators_data.csv`_.

It is enabled by a value in the configuration tree::

  costing:
    GenXInterface: <bool>

.. _GenX: https://genxproject.github.io/GenX/dev/

.. _generators_data.csv: https://genxproject.github.io/GenX/dev/data_documentation/#.1.5-Generators_data.csv
.. currentmodule:: faroes.generomakcosting

.. autoclass:: GeneromakToGenX

.. footbibliography::
