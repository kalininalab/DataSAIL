#################
Supported Solvers
#################

.. _solver-label:

With the default installation settings, DataSAIL supports SCIP. Both solvers are free to use. Additional
solvers are supported if the user installs them. All installation steps shall be executed in the conda-environment that
also contains DataSAIL.

CPLEX
=====

CPLEX is a solver developed by IBM and can be installed from conda by executing

.. code-block:: shell

    mamba install -c ibmdecisionoptimization cplex

Then, you can request the CPLEX solver by :code:`--solver CPLEX` (CLI) or :code:`solver="CPLEX"` (Python API).
To use CPLEX, you need to have a valid license. You can request a free academic license from IBM.

GLPK_MI
=======

GLPK is a solver developed by Andrew Makhorin and can be installed from conda by executing

.. code-block:: shell

    mamba install -c conda-forge glpk

Then, you can request the GLPK solver by :code:`--solver GLPK` (CLI) or :code:`solver="GLPK"` (Python API).
GLPK is free to use but did not perform well in our tests. Therefore, we recommend using SCIP instead.

Gurobi
======

Gurobi is a solver developed by Gurobi Optimization and can be installed from conda by executing

.. code-block:: shell

    mamba install -c gurobi gurobi

Then, you can request the Gurobi solver by :code:`--solver GUROBI` (CLI) or :code:`solver="GUROBI"` (Python API).
To use Gurobi, you need to have a valid license. You can request a free academic license from Gurobi.

MOSEK
=====

MOSEK is a solver developed by MOSEK ApS and can be installed from conda by executing

.. code-block:: shell

    mamba install -c mosek mosek

Then, you can request the MOSEK solver by :code:`--solver MOSEK` (CLI) or :code:`solver="MOSEK"` (Python API).
To use MOSEK, you need to have a valid license. You can request a free academic license from MOSEK.

SCIP
====

SCIP is a solver developed by Zuse Institute Berlin and can be installed from conda by executing

.. code-block:: shell

    mamba install -c conda-forge pyscipopt

Then, you can request the SCIP solver by :code:`--solver SCIP` (CLI) or :code:`solver="SCIP"` (Python API).

XPRESS
======

XPRESS is a solver developed by FICO and can be installed from conda by executing

.. code-block:: shell

    mamba install -c fico xpress

Then, you can request the XPRESS solver by :code:`--solver XPRESS` (CLI) or :code:`solver="XPRESS"` (Python API).
To use XPRESS, you need to have a valid license. You can request a free academic license from FICO.
