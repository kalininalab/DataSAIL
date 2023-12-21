#################
Supported Solvers
#################

.. _solver-label:

With the default installation settings, DataSAIL supports SCIP. This is a free solver that performed best among the
free solvers during development. Additional solvers are supported upon user installation. All installation steps shall
be executed in the conda-environment that also contains DataSAIL. Below, we describe the installation steps for the
supported solvers.

.. note::
    The installation of the solvers is not part of the DataSAIL installation. We provide the installation steps here
    for convenience. If you encounter any problems during the installation, please refer to the documentation of the
    solver.

.. note::
    In our tests during development, we mostly used SCIP, MOSEK, and GUROBI due to convenience. In favour of runtime,
    we advice to use GUROBI even though licensing can be a bit more complex.

Main Solvers
############

Our top suggestion is to use GUROBI. It is a commercial solver, but offers a free academic license. It showed to be the
fastest and most accurate solver during development. If you do not want to use a commercial solver, we suggest to use
SCIP. It is a free solver, but slower than GUROBI (factor 2-3). MOSEK is a commercial solver but offers a free academic
too. Its runtime ranges between SCIP and GUROBI.

GUROBI
======

GUROBI is a solver developed by Gurobi Optimization and can be installed from conda by executing

.. code-block:: shell

    mamba install -c gurobi gurobi

Then, you can request the GUROBI solver by :code:`--solver GUROBI` (CLI) or :code:`solver="GUROBI"` (Python API).
To use GUROBI, you need to have a valid license. You can request a free academic license from the
`GUROBI website <https://www.gurobi.com/features/academic-named-user-license/>`_. make sure that the license covers
your installed version of GUROBI.

SCIP
====

SCIP is a solver developed by Zuse Institute Berlin and can be installed from conda by executing

.. code-block:: shell

    mamba install -c conda-forge pyscipopt

Then, you can request the SCIP solver by :code:`--solver SCIP` (CLI) or :code:`solver="SCIP"` (Python API).

MOSEK
=====

MOSEK is a solver developed by MOSEK ApS and can be installed from conda by executing

.. code-block:: shell

    mamba install -c mosek mosek

Then, you can request the MOSEK solver by :code:`--solver MOSEK` (CLI) or :code:`solver="MOSEK"` (Python API).
To use MOSEK, you need to have a valid license. You can request a free academic license from the
`MOSEK website <https://www.mosek.com/products/academic-licenses/>`_.

Additional Solvers
##################

The following solvers are offers for convenience. We did not use them during development and for different reasons.
CBC and GLPK_MI are free solvers and can be used without any license, but we had issues running them on larger problem
instances. CPLEX and XPRESS are commercial solvers and we did not manage to get the license to work properly.

CBC
===

CBC is a solver developed by the COIN-OR foundation and can be installed from conda by executing

.. code-block:: shell

    mamba install -c conda-forge coin-or-cbc
    pip install cylp

Then, you can request the SCIP solver by :code:`--solver CBC` (CLI) or :code:`solver="CBC"` (Python API). CBC is a free
solver and can be used without any license.

CPLEX
=====

CPLEX is a solver developed by IBM and can be installed from conda by executing

.. code-block:: shell

    mamba install -c ibmdecisionoptimization cplex

Then, you can request the CPLEX solver by :code:`--solver CPLEX` (CLI) or :code:`solver="CPLEX"` (Python API).
To use CPLEX, you need to have a valid license. You can request a free academic license from the
`IBM website <https://ampl.com/products/solvers/solvers-we-sell/cplex/>`_.

GLPK_MI
=======

GLPK is a solver developed by Andrew Makhorin and can be installed from conda by executing

.. code-block:: shell

    mamba install -c conda-forge cvxopt

Then, you can request the GLPK_MI solver either by :code:`--solver GLPK` or :code:`--solver GLPK_MI` (both for CLI) or
:code:`solver="GLPK"` or :code:`solver="GLPK_MI"` (both Python API). Technically, GLPK and GLPK_MI are two different
solver, but developed by the same group and GLPK_MI is an extension of GLPK for mixed-integer problems. Because
GLPK(_MI) is part of the "GNU universe" is free to use.

XPRESS
======

XPRESS is a solver developed by FICO and can be installed from conda by executing

.. code-block:: shell

    mamba install -c fico-xpress xpress

Then, you can request the XPRESS solver by :code:`--solver XPRESS` (CLI) or :code:`solver="XPRESS"` (Python API).
To use XPRESS, you need to have a valid license. You can request a free academic license from FICO.
