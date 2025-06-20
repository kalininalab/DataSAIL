.. _cli-label:

#####################
Commandline Interface
#####################

Here, we discuss the arguments for the Commandline Interface in more detail. As they are more or less the same as for
the package usage of DataSAIL, this is also an extended description of package.

General Arguments
#################

In this section, we're discussing the argument structure of DataSAIL. The arguments are mostly the same between the
python function and the CLI. Their functionality does not change, but some of the arguments are not available for the
package version. This is noted accordingly. What might change is the type of input accepted. The package version of
DataSAIL usually accepts string input to a file, a dictionary or a list (depending on the argument), and a function or
generator therefore. For more details on the supported types, please checkout the type annotations of the
:ref:`package entry to DataSAIL <doc-label>`.

-o / -\-output
==============
CLI only! Required!

The path to the output directory to store the splits in. This folder will contain all splits, reports, and logs from
the execution.

-i / -\-inter
=============
The filepath to the TSV file of interactions between two entities. More details are given :ref:`here <files-xsv-label>`.

-\-to-sec
=========
The maximal time to spend optimizing the objective in seconds. This does not include preparatory work such as parsing
data and clustering the input.

-\-threads
==========
The number of threads to use throughout the computation. This number of threads is also forwarded to clustering
programs used internally. If 0, all available CPUs will be used.

-\-verbose
==========
The verbosity level of the program. Choices are: [C]ritical, [F]atal, [E]rror, [W]arning, [I]nfo, [D]ebug

-v / -\-version
===============
CLI only!

Get the number of the installed version of DataSAIL.

Splitting Arguments
###################

The following arguments are used to specify the splitting mode and the splits to compute. The arguments are the same
for the CLI and the package version of DataSAIL.

-t / -\-techniques
==================
Required!

Select the mode to split the data. Choices are
  * R: Random split,
  * I1: identity-based cold-single split,
  * I2: identity-based cold-double split,
  * C1: similarity-based cold-single split,
  * C2: similarity-based cold-double split

For both, I1 and C1, you have to specify e or f, i.e. I1e, I1f, C1e, or C1f, to make clear if DataSAIL shall
compute a cold split based on the e-entity or the f-entity.

-s / -\-splits
==============
The sizes of the individual splits the program shall produce.

-n / -\-names
=============
The names of the splits in order of the -s argument. If left empty, splits will be called Split1, Split2, ...

-\-overflow
===========
How to handle overflow of the splits. If 'assign', a cluster that overflows a split size will be assigned to one split. 
The remaining data is split normally into n-1 splits. If 'break', the cluster will be broken into smaller parts to fit into a split.

-d / -\-delta
=============
A multiplicative factor by how much the limits (as defined in the -s / --splits argument defined) of the stratification
can be exceeded.

-e / -\-epsilon
===============
A multiplicative factor by how much the limits (as defined in the -s / --splits argument defined) of the splits can be
exceeded.

-r / -\-runs
============
The number of different to perform per technique. The idea is to compute several different splits of the dataset using
the same technique to investigate the variance of the model on different data-splits. The variance in splits is
introduced by shuffling the dataset everytime a new split is requested.

-\-solver
=========
Which solver to use to solve the binary linear program. The choices are presented :ref:`here <solver-label>`.

-\-cache
========
Boolean flag indicating to store clustering matrices in cache to not recompute clusters multiple times.

-\-cache-dir
============
Destination of the cache folder. Default is the OS-default cache dir

Entity Arguments
################

The following arguments are entity specific and the same for e entities and f entities. We will describe the arguments
for the e entities. The arguments for the f entities can be derived by replacing "e-" with "f-".

-\-e-type
=========
The type of the first data batch to the program. Choices are: [P]rotein, [M]olecule, [G]enome, [O]ther"

-\-e-data
=========
The first input to the program. This can either be the filepath a directory containing only data files.

-\-e-weights
============
The custom weights of the samples, the format can be a :ref:`CSV/TSV-file <files-xsv-label>` or equivalent as described
above.

-\-e-sim
========
Provide the name of a method to determine similarity between samples of the first input dataset. This can either be the
name of a method based on the data type (see :ref:`here <clustering-label>` for available methods) or a filepath to a
file storing the pairwise similarities in TSV (see :ref:`here <files-xsv-label>` for details).

-\-e-dist
=========
Provide the name of a method to determine distance between samples of the first input dataset. This can either be the
name of a method based on the data type (see :ref:`here <clustering-label>` for available methods) or a filepath to a
file storing the pairwise similarities in TSV (see :ref:`here <files-xsv-label>` for details).

-\-e-strat
==========
A file containing the stratification of the first input dataset. The stratification is a TSV file as described
:ref:`here <files-xsv-label>`.

-\-e-args
=========
Additional arguments for the clustering algorithm used in -\-e-dist or -\-e-sim.
