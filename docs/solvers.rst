#################
Supported Solvers
#################

With the default installation settings, DataSAIL supports SCIP and GLPK_MI. Both solvers are free to use. Additional
solvers are supported if the user installs them. All installation steps shall be executed in the conda-environment that
also contains DataSAIL.

CPLEX
=====

CPLEX is a solver developed by IBM and can be installed from conda by executing

.. code-block:: shell

    mamba install -c ibmdecisionoptimization cplex

Then, you can requets the CPLEX solver by --solver CPLEX or solver="CPLEX"