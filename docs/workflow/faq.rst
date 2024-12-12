################################
Frequently Asked Questions (FAQ)
################################

BLAS
----

.. code:: shell

    BLAS : Program is Terminated. Because you tried to allocate too many memory regions.
    Segmentation fault

This is a problem related to OpenBLAS used within sklean (see the OpenBLAS FAQ here:
https://github.com/OpenMathLib/OpenBLAS/wiki/faq#allocmorebuffers. The solution is to execute the following code
snippet BEFORE IMPORTING :code:`datasail`:

.. code:: python

    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["GOTO_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

