**********
Clustering
**********

Clustering is an essential step in DataSAIL as it groups similar samples together such that they cannot be assigned to
different splits. This reduces the information leakage even more. In the following table you can see which clustering
algorithms are available, what output they produce and what input they take for which type of data.

By the term `clustering` in DataSAIL, we understand two things. Either group samples together (what you usually mean
when talking about clustering). But also computing a pairwise similarity or distance matrix for the samples is a form
of clustering in DataSAIL as these matrices can be used in Affinity Propagation or Agglomerative Clustering (summarized
in the term "additional clustering") to compute clusters based on the precomputed similarities or distances.

Usually in DataSAIL, clustering is performed in multiple rounds and there are two options for this:

#. Either, DataSAIL computes a pairwise similarity or distance matrix for your input data. Then, DataSAIL performs
   additional clustering until the number of clusters cannot be further reduced or reaches a window such that the
   problem for the constraint optimizer is feasible in reasonable time.
#. Or DataSAIL uses a clustering algorithm that does not return a pairwise matrix (CD-HIT or MMseqs2). Then, the
   parameters of the algorithm are tweaked using binary search to find a set of parameters that results in the minimal
   number of clusters or reaches a window as described in 1.

Overview
--------

The following table shows an overview over the different input types and which clustering algorithms are available.
The ability to cluster "other" data (such as Ferrari cars) is a side effect of the implementation. It is listed in the
table despite there is no clustering algorithm for this. Therefore, the only way to determine similarities or distances
of entities of "other" input type is to provide them as a matrix in a file and then apply additional clustering based
on these matrices.

.. list-table:: Input molecule types and their available clustering algorithms
    :widths: 25 20 15 15 15 15 15
    :header-rows: 1

    * - Clust. Algo
      - Protein
      - Molecule
      - Genomic
      - Other
      - Sim or Dist
      - Inter-clust
    * - CD-HIT
      - FASTA
      - \-
      - \-
      - \-
      - Sim
      - No
    * - Scaffold + ECFP + Tanimoto Coeffs
      - \-
      - SMILES
      - \-
      - \-
      - Sim
      - Yes
    * - FoldSeek
      - PDB
      - \-
      - \-
      - \-
      - Sim
      - Yes
    * - File input
      - file
      - file
      - file
      - file
      - Both
      - Yes
    * - MASH
      - \-
      - \-
      - FASTA
      - \-
      - Dist
      - Yes
    * - MMseqs2
      - FASTA
      - \-
      - \-
      - \-
      - Sim
      - No
    * - WLKernel
      - PDB
      - SMILES
      - \-
      - \-
      - Sim
      - Yes

The last column tells if a certain clustering algorithm produces a pairwise distance or similarity matrix or not. In
the former case, rounds of additional clustering are conducted next. In case of no pairwise matrix, the parameters of
the algorithm are optimized until the best clustering of the data for DataSAIL has been found.

Default Algorithms
------------------

To simplify the task of algorithm and parameter selection and for the (inexperienced) user, DataSAIL provides default
algorithms and arguments for each type of input.

.. list-table:: Default algorithms based on the type of data and the format of data
    :header-rows: 1

    * -
      - Protein
      - Molecule
      - Genomic
      - Other
    * - PDB
      - FoldSeek
      -
      -
      -
    * - FASTA
      - CD-HIT
      -
      - MASH
      -
    * - TSV
      -
      - ECFP++
      -
      -

Details about the clustering algorithms
=======================================

CD-HIT
------

CD-HIT is used to cluster protein sequences, for more information on CD-HIT, visit the `website <https://sites.google.com/view/cd-hit>`_,
checkout the `GitHub repository <https://github.com/weizhongli/cdhit>`_, or read the `paper <https://doi.org/10.1093/bioinformatics/bts565>`_.

CD-HIT has two parameters to adjust how fine or coarse the clustering will be. Those are :code:`-n` and :code:`-c`.
Those are automatically adjusted and searched to find a good clustering to start splitting the data.

The general command to run CD-HIT in DataSAIL is

.. code-block:: shell

    cd-hit -i <input> -o clusters -g 1 -n ? -c ?

where the values for :code:`-n` and :code:`-c` are optimized as described above.

ECFP++
------

ECFP++ is a short name for a 3-step process to detect clusters in a dataset of chemical molecules. The first step is to
compute Scaffolds following `RDKits MakeScaffoldGeneric <https://rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html#rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric>`_.
This way, molecules are simplified by replacing every heavy atom with carbon atoms and every bond with a single bond.
The second step is to compute a 1024-bit `Morgan fingerprint <https://doi.org/10.1021/ci100050t>`_ with radius 2.
Lastly, DataSAIL computes the similarity of these fingerprints as `Tanimoto-Similarities <https://en.wikipedia.org/wiki/Jaccard_index>`_
of the bit-vectors.

FoldSeek
--------

FoldSeek is used to cluster protein structures based on PDB input. For more information checkout the `GitHub repository <https://github.com/steineggerlab/foldseek>`_
and the `paper <https://doi.org/10.1101/2022.02.07.479398>`_.

As FoldSeek produces a pairwise similarity matrix, it is not optimizes such as CD-HIT, but will be followed by some
additional clustering.

The general command to run FoldSeek in DataSAIL is

.. code-block:: shell

    foldseek easy-search <pdb_dir> <pdb_dir> aln.m8 tmp --alignment-type 1 --tmscore-threshold 0.0 --format-output 'query,target,fident' --exhaustive-search 1 -e inf

MASH
----

To cluster genomes in DataSAIL, the only option so far is MASH (CD-HIT-EST is to be included). Similar to FoldSeek it
produces a pairwise distance matrix which is used in subsequent rounds of additional clustering. To get more
information on MASH, read the `paper <https://doi.org/10.1186/s13059-016-0997-x>`_ and the `ReadTheDocs page <https://mash.readthedocs.io/en/latest/>`_.

DataSAIl calls MASH in two steps. First to compute the sketches and then to compute their distance

.. code-block:: shell

    mash sketch -s 10000 -o ./cluster input
    mash dist -t cluster.msh cluster.msh > cluster.tsv

MMseqs2
=======

An alternative to CD-HIT to cluster protein sequences is MMseqs2. To get more information on the functionality of
MMseqs2, checkout the `GitHub repository <https://github.com/soedinglab/MMseqs2>`_ and the `paper <https://doi.org/10.1038/nbt.3988>`_.

To interact with MMseqs2, DataSAIL calls it through commandline with

.. code-block:: shell

    mmseqs easy-cluster <input> mmseqs_out mmseqs_tmp --similarity-type 2 --cov-mode 0 -c 0.8 --min-seq-id ?

Like CD-HIT, MMseqs2 does not output pairwise similarities, therefore, a sequence similarity parameter has to be
tweaked to find the best clustering for DataSAIL to work with. The parameter in question here is :code:`--min-seq-id`.

WL-Kernel
---------

The last method to compute similarities of graph-structured data such as PDB files is to use Weisfeiler-Lehman kernels.
This method is not established and mostly experimental, therefore there is no literature to link, but you can have a
look at `grakel <https://ysig.github.io/GraKeL/0.1a8/>`_, the Python package DataSAIL uses to apply WLKernel.
