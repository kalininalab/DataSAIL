########
DataSAIL
########

Data Splitting Against Information Leakage
The code is available onf `GitHub <https://github.com/kalininalab/datasail>`_ and can be installed from
`conda <https://anaconda.org/kalininalab/datasail>`_.

Quick Start
===========

Other than described on the conda-website, the command to install DataSAIL is

.. code-block:: shell

    conda install -c kalininalab -c conda-forge -c mosek datasail
    pip install grakel

The second command is necessary to run WLK clustering as the grakel library is not available on conda for python 3.10.

The usage is pretty simple, just execute

.. code-block:: shell

    datasail -h

to run DataSAIL in the command line and see the parameters DataSAIL takes. For a more detailed description see
:ref:`here <datasail-doc-label>`. DataSAIL can also directly be included as a normal package into your Python program
using

.. code-block:: python

    from datasail.sail import datasail

    splits = datasail(...)

The arguments for the package use of DataSAIL are explained HERE and HERE you can find a more detailed description of
them based on their CLI use as the arguments are mostly the same.

**********************
DataSAIL documentation
**********************

Welcome to the documentation of DataSAIL!

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Workflow

    workflow/input
    workflow/clustering
    workflow/splits

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Interfaces

    interfaces/cli
    interfaces/package
