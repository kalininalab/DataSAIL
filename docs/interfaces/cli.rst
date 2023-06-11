*********************
Commandline Interface
*********************

.. _datasail-cli-label:

Here, we discuss the arguments for the Commandline Interface in more detail. As they are more or less the same as for
the package usage of DataSAIL, this is also an extended description of package.

Arguments
=========

In this section, we're discussing the argument structure of DataSAIL. The arguments are mostly the same between the
python function and the CLI. Their functionality does not change, but some of the arguments are not available for the
package version. This is noted accordingly. What might change is the type of input accepted. The package version of
DataSAIL usually accepts string input to a file, a dictionary or a list (depending on the argument), and a function or
generator therefore. For more details on the supported types, please checkout the type annotations of the
:ref:`package entry to DataSAIL <datasail-doc-label>`.

**Every TSV file has to have a header line. Otherwise, the first line entry is ignored by DataSAIL.**

-o / -\-output
--------------
CLI only! Required!

The path to the output directory to store the splits in. This folder will contain all splits, reports, and logs from the
execution.

-i / -\-inter
-------------
The filepath to the TSV file of interactions between two entities. The first entry in each line has to match an entry
from the e-entity, the second matches one of the f-entity. You can specify an interaction file even-though you don't
specify both types of entities. In case, interaction are provided they are used to compute weights for both entities.

-\-to-sec
---------
The maximal time to spend optimizing the objective in seconds. This does not include preparatory work such as parsing
data and clustering the input.

-\-to-sol
---------
The maximal number of solutions to compute until end of search (in case no optimum was found). This argument is ignored
so far.

-\-threads
----------
The number of threads to use throughout the computation. This number of threads is also forwarded to clustering programs
used internally. If 0, all available CPUs will be used.

-\-verbose
----------
The verbosity level of the program. Choices are: [C]ritical, [F]atal, [E]rror, [W]arning, [I]nfo, [D]ebug

-v / -\-version
---------------
CLI only!

Get the number of the installed version of DataSAIL.

-t / -\-techniques
------------------
Required!

Select the mode to split the data. Choices are
  * R: Random split,
  * ICS: identity-based cold-single split,
  * ICD: identity-based cold-double split,
  * CCS: similarity-based cold-single split,
  * CCD: similarity-based cold-double split

For both, ICS and CCS, you have to specify e or f, i.e. ICSe, ICSf, CCSe, or CCSf, to make clear if DataSAIL shall
compute a cold split based on the e-entity or the f-entity.

-s / -\-splits
--------------
The sizes of the individual splits the program shall produce.

-n / -\-names
-------------
The names of the splits in order of the -s argument. If left empty, splits will be called Split1, Split2, ...

-e / -\-epsilon
---------------
A multiplicative factor by how much the limits (as defined in the -s / --splits argument defined) of the splits can be
exceeded.

-r / -\-runs
------------
The number of different to perform per technique. The idea is to compute several different splits of the dataset using
the same technique to investigate the variance of the model on different data-splits. The variance in splits is
introduced by shuffling the dataset everytime a new split is requested.

-\-solver
---------
Which solver to use to solve the binary quadratic program. Choices are SCIP (free of charge) and MOSEK (licensed and
only applicable if a valid mosek license is stored (see the `MOSEK website <https://www.mosek.com/>`__ for more
information on licensing). Note: You can still use the program, even if you don't have a MOSEK license and rely on SCIP.

-\-scalar
---------
A boolean flag indicating to run the program in scalar for instead of vectorized formulation [default]."

-\-cache
--------
Boolean flag indicating to store clustering matrices in cache to not recompute clusters multiple times.

-\-cache-dir
------------
Destination of the cache folder. Default is the OS-default cache dir


The following arguments are entity specific and the same for e entities and f entities. We will describe the arguments
for the e entities. The arguments for the f entities can be derived by replacing "e-" with "f-".

-\-e-type
---------
The type of the first data batch to the program. Choices are: [P]rotein, [M]olecule, [G]enome, [O]ther"

-\-e-data
---------
The first input to the program. This can either be the filepath a directory containing only data files.

-\-e-weights
-------------
The custom weights of the samples. The file has to have TSV format where every line is of the form [e_id >tab< weight].
The e_id has to match an entity id from the first input argument.

-\-e-sim
--------
Provide the name of a method to determine similarity between samples of the first input dataset. This can either be
cdhit, ecfp, foldseek, mmseqs, wlk, or a filepath to a file storing the pairwise similarities in TSV.

-\-e-dist
---------
Provide the name of a method to determine distance between samples of the first input dataset. This can be MASH or a
filepath to a file storing the pairwise distances in TSV.

-\-e-args
---------
Additional arguments for the clustering algorithm used in -\-e-dist or -\-e-sim.

-\-e-max-sim
------------
The maximum similarity of two samples from the first data in two split.

-\-e-max-dist
-------------
The maximal distance of two samples from the second data in the same split.
