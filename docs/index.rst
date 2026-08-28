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

    If you want to collaborate with us on using DataSAIL on non-biochemical datasets, please reach out to us via email
    at :code:`roman.joeres[at]helmholtz-hips.de`.
    

Install
#######

.. note::

    **This installation instruction changed in version 1.4.0 compared to previous versions** as we have merged the :code:`datasail` and :code:`datasail-lite` packages into a single one. The usage remains the same, both as a python package and as a commandline tool.
    
    From version 1.4.0 onwards, DataSAIL comes without third-party clustering algorithms such as MMseqs2, CD-HIT, FoldSeek or MASH. If you want to use these tools, please install them separately and make sure they are in your PATH. 
    For more information on how to install these tools, please see the :ref:`Section on clustering algorithms <clustering-algorithms>` in the documentation.

DataSAIL is available for all modern versions of Python (v3.10 or newer). You can install DataSAIL using either pip (recommended)

.. code-block:: shell

    pip install datasail

or `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ (fast alternative of conda) with the following command:

.. code-block:: shell

    mamba install -c conda-forge -c kalininalab datasail

By default, DataSAIL v1.4 installs NumPy v2. If you want to use DataSAIL with NumPy v1, please append :code:`numpy<2` to the installation command.
For information on how to install DataSAIL v1.3 and older, please refer to the :ref:`old installation <faq-old-installation>` section in the FAQ.

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
