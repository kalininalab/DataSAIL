########
DataSAIL
########

The code for Data Splitting Against Information Leakage, short DataSAIL, is available on
`GitHub <https://github.com/kalininalab/datasail>`_ and can be installed from
`conda <https://anaconda.org/kalininalab/datasail>`_ using
`mamba <https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install>`_.

Quick Start
===========

DataSAIL is avalable for all modern versions of Pytion (v3.8 or newer). Other than described on the conda-website, 
the command to install DataSAIL within your just created environment is

.. code-block:: shell

    mamba install -c kalininalab -c mosek -c conda-forge -c bioconda datasail
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

Usability
=========

DataSAIL is available for Linux, MacOS, and Windows. As DataSAIL relies on MMMseqs2, MASH, and FoldSeek, this cannot 
be used on Windows. Therefore, DataSAIL on Windows is limited to ECFP++ and user-specific datatypes.

DataSAIL is avaliable for all modern Python versions, i.e., 3.8-3.12.

.. toctree::
    :maxdepth: 3
    :caption: Workflow

    workflow/input
    workflow/clustering
    workflow/splits
    workflow/solvers

.. toctree::
    :maxdepth: 0
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

