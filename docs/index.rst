########
DataSAIL
########

DataSAIL, short for Data Splitting Against Information Leakage, serves as a tool crafted to partition data in a manner
that minimizes information leakage, especially tailored for machine learning workflows dealing with biological
datasets. However, its versatility extends beyond biology, making it applicable to various types of datasets. Whether
utilized through its command line interface or integrated as a Python package, DataSAIL stands out for its
user-friendly design and adaptability. Licensed under the MIT license, it is open source and conveniently accessible on
`GitHub <https://github.com/kalininalab/datasail>`_. Installation is made simple through
`conda <https://anaconda.org/kalininalab/datasail>`_, utilizing
`mamba <https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install>`_.

Quick Start
###########

DataSAIL is available for all modern versions of Pytion (v3.8 or newer). Other than described on the conda-website,
the command to install DataSAIL within your just created environment is

.. code-block:: shell

    mamba install -c kalininalab -c conda-forge -c bioconda datasail
    pip install grakel

The second command is necessary to run WLK clustering as the grakel library is not available on conda for python 3.10
or newer. Alternatively, one can install :code:`DataSAIL-lite` from conda as

.. code-block:: shell

    mamba install -c kalininalab -c conda-forge -c bioconda datasail-lite
    pip install grakel

.. note::
    It is important to use mamba for the installation because conda might not be able to resolve the dependencies of
    DataSAIL successfully.

The difference between :code:`DataSAIL` and :code:`DataSAIL-lite` is that the latter does not include the clustering
algorithms and requires the user to install them manually as needed. The reason for this is that the clustering
algorithms are not available for all OS and we want to make DataSAIL available for all OS.

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

