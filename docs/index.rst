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

DataSAIL is available for all modern versions of Python (v3.9 or newer). We ship two versions of DataSAIL:

- :code:`DataSAIL`: The full version of DataSAIL, which includes all third-party clustering algorithms and is available on conda for linux and OSX (called :code:`datasail`).

- :code:`DataSAIL-lite`: A lightweight version of DataSAIL, which does not include any third-party clustering algorithms and is available on PyPI (called :code:`datasail`) and conda (called :code:`datasail-lite`).

.. note::
    There is a naming-inconsitency between the conda and PyPI versions of DataSAIL. The lite version is called :code:`datasail-lite` on conda, while it is called :code:`datasail` on PyPI. 
    This will be fixed in the future, but for now, please be aware of this inconsistency.

.. raw:: html
    :file: install.html

.. note::
    If you install DataSAIL from conda, it is recommended to use `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_
    because conda might not be able to resolve the dependencies of DataSAIL successfully.

Quick Start
###########

Regardless of which installation command was used, DataSAIL can be executed by running

.. code-block:: shell

    datasail -h

in the command line and see the parameters DataSAIL takes. For a more detailed description see
:ref:`here <doc-label>`. DataSAIL can also directly be included as a normal package into your Python program
using

.. code-block:: python

    from datasail.sail import datasail

    splits = datasail(...)

The arguments for the package use of DataSAIL are explained in the :ref:`method's documentation <doc-label>`.
You can find a more detailed description of them based on their :ref:`CLI <cli-label>` use as the
arguments are mostly the same.

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
    posters
