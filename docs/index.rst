########
DataSAIL
########

DataSAIL, short for Data Splitting Against Information Leakage, serves as a tool crafted to partition data in a manner
that minimizes information leakage, especially tailored for machine learning workflows dealing with biological
datasets. However, its versatility extends beyond biology, making it applicable to various types of datasets. Whether
utilized through its command line interface or integrated as a Python package, DataSAIL stands out for its
user-friendly design and adaptability. Licensed under the MIT license, it is open source and conveniently accessible on
`GitHub <https://github.com/kalininalab/datasail>`_. Installation is made simple through
`conda <https://anaconda.org/kalininalab/datasail>`_.

Install
#######

DataSAIL is available for all modern versions of Pytion (v3.9 or newer).

.. note::
    It is recommended to use `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_
    for the installation because conda might not be able to resolve the dependencies of DataSAIL successfully.

.. raw:: html
    :file: install.html

DataSAIL vs. DataSAIL-lite
--------------------------

The difference between :code:`DataSAIL` and :code:`DataSAIL-lite` is that the latter does not include most of the
clustering algorithms as they are not provide on conda for all OSs. Therefore, the user is required to the user to
install them manually as needed. DataSAIL will work even if not all clustering are installed. For the installation, is
it necessary to be able to call them. You can test which are available by running :code:`datasail --cc`.

Quick Start
###########

Regardless of which installation command was used, DataSAIL can be executed by running

.. code-block:: shell

    datasail -h

in the command line and see the parameters DataSAIL takes. For a more detailed description see
:ref:`here <datasail-doc-label>`. DataSAIL can also directly be included as a normal package into your Python program
using

.. code-block:: python

    from datasail.sail import datasail

    splits = datasail(...)

The arguments for the package use of DataSAIL are explained in the :ref:`method's documentation <datasail-doc-label>`.
You can find a more detailed description of them based on their :ref:`CLI <datasail-cli-label>` use as the
arguments are mostly the same.

.. toctree::
    :maxdepth: 1
    :caption: Workflow

    workflow/input
    workflow/clustering
    workflow/embeddings
    workflow/splits
    workflow/solvers
    posters

.. toctree::
    :maxdepth: 1
    :caption: Interfaces

    interfaces/cli
    interfaces/package

.. toctree::
    :maxdepth: 1
    :caption: Examples

    examples/qm9
    examples/bace
    examples/pdbbind
    examples/rna
    examples/tox21
    examples/asteroids

.. toctree::
    :maxdepth: 1
    :caption: Extend DataSAIL

    extensions/contributing
    extensions/metric

