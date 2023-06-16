########
DataSAIL
########

The code for Data Splitting Against Information Leakage, short DataSAIL, is available on
`GitHub <https://github.com/kalininalab/datasail>`_ and can be installed from
`conda <https://anaconda.org/kalininalab/datasail>`_ using :code:`mamba`.

Quick Start
===========

DataSAIL currently only runs in Python 3.10. Therefore, you have to install it into a Python 3.10 environment. For
conda, this can be created by running

.. code-block:: shell

    conda create -n datasail python=3.10

Other than described on the conda-website, the command to install DataSAIL within your just created environment is

.. code-block:: shell

    mamba install -c kalininalab -c conda-forge -c mosek -c bioconda datasail
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

The arguments for the package use of DataSAIL are explained in the :ref:`methods documentation <datasail-doc-label>`.
You can find a more detailed description of them based on their :ref:`CLI <datasail-cli-label>` use as the
arguments are mostly the same.

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
