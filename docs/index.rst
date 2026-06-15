########
DataSAIL
########

DataSAIL, short for Data Splitting Against Information Leakage, is a versatile tool designed to partition data while 
minimizing similarities between the partitions. Inter-sample similarities can lead to information leakage, resulting 
in an overestimation of the model's performance in certain training regimes.

DataSAIL was initially developed for machine learning workflows involving biological datasets, but its utility extends to
any type of datasets. It can be used through a command line interface or integrated as a Python package, making it
accessible and user-friendly. The tool is licensed under the MIT license, ensuring it remains open source and freely
available on `GitHub <https://github.com/kalininalab/datasail>`_.

.. note::

    DataSAIL is a work in progress, and we are continuously improving it. If you have any suggestions or find any bugs,
    please open an issue in our `Issue Tracker <https://github.com/kalininalab/datasail/issues>`_ on GitHub.

.. note::

    If you want to collaborate with us on using DataSAIL on non-biochemical datasets, please reach out to us via email
    at :code:`roman.joeres[at]helmholtz-hips.de`.
    

Install
#######

.. note::

    This installation instruction changed in version 1.4.0 compared to previous versions. The usage remains the same, both as a python package and as a commandline tool.

DataSAIL is available for all modern versions of Python (v3.10 or newer). You can install DataSAIL using either pip (recommended)

.. code-block:: shell

    pip install datasail

or conda/mamba

.. code-block:: shell

    mamba install -c conda-forge -c kalininalab datasail

.. note::
    
    This installation instruction changed in version 1.4.0 compared to previous versions. The usage remains the same, both as a python package and as a commandline tool.

Until version 1.3.0, DataSAIL was available in two versions. From version 1.4.0 onwards, we have merged the two versions into a single one. Both verions come without third-party clustering algorithms such as MMseqs2, CD-HIT, FoldSeek or MASH. If you want to use these tools, please install them separately and make sure they are in your PATH. For more information on how to install these tools, please see the [documentation page](https://datasail.readthedocs.io/en/latest/installation.html#external-clustering-tools).

.. note::
    
    If you install DataSAIL from conda, it is recommended to use `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_
    because conda might not be able to resolve the dependencies of DataSAIL successfully.

By default, DataSAIL v1.4 installs NumPy v2. If you want to use DataSAIL with NumPy v1, please use 

.. code-block:: shell

    pip install datasail numpy<2

for installing from the PyPI or

.. code-block:: shell

    mamba install -c conda-forge -c kalininalab datasail numpy<2

to install from conda/mamba.

Quick start
###########

DataSAIL comes with a command-line interface and as a Python package. The main functionality can be accessed by, e.g., running the following command in the terminal: 

.. code-block:: shell

    datasail --output <path_to_output_path> --technique C1e --e-type P --e-data <path_to_fasta> --e-sim mmseqs

or in a Python program by

.. code-block:: python

    from datasail.sail import datasail

    splits = datasail(technique=["C1e"], e_type="P", e_data="<path_to_fasta>", e_sim="mmseqs", output="<path_to_output_path>")

Here, the output argument is optional and saves the results in a folder in addition to returning them. For more information about the parameters, please read through the :ref:`documentation page <doc-label>`.

FAQ
###

For frequently asked questions, please refer to the :ref:`FAQ <faq-label>` section.

.. toctree::
    :maxdepth: 1
    :caption: Workflow

    workflow/workflow
    workflow/input
    workflow/clustering
    workflow/embeddings
    workflow/solvers
    workflow/splits

.. toctree::
    :maxdepth: 1
    :caption: Interfaces

    interfaces/cli
    interfaces/package
    interfaces/dl_eval

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

.. toctree::
    :maxdepth: 1
    :caption: Miscellaneous
    
    faq
    other
    posters
