Welcome to the documentation of DataSAIL
========================================

Data Splitting Against Information Leakage
The code is available onf github: https://github.com/kalininalab/datasail
and can be installed with `pip`: https://pypi.org/project/datasail/ (not yet available)

DataSAIL documentation
======================

Welcome to the documentation of DataSAIL!

Clustering
----------

The following table shows an overview over the different input types and which clustering algorithms are available.
The ability to cluster "other" data (such as Ferrari cars) is a side effect of the implementation. It is listed in the
table despite there is no clustering algorithm for this. Therefore, the only way to determine similarities or distances
of entities of "other" input type is to provide them as a matrix in a file.

.. list-table:: Input molecule types and their available clustering algorithms
    :widths: 25 15 15 15 15 15 15
    :header-rows: 1

    * - Clust. Algo
      - Sim or Dist
      - Protein
      - Molecule
      - Genomic
      - Other
      - Inter-clust
    * - CD-HIT
      - Sim
      - FASTA
      - \-
      - \-
      - \-
      - No
    * - ECFP + Tanimoto
      - Sim
      - \-
      - SMILES
      - \-
      - \-
      - Yes
    * - FoldSeek
      - Sim
      - PDB
      - \-
      - \-
      - \-
      - Yes
    * - File input
      - Both
      - file
      - file
      - file
      - file
      - Yes
    * - MASH
      - Dist
      - \-
      - \-
      - FASTA
      - \-
      - Yes
    * - MMseqs2
      - Sim
      - FASTA
      - \-
      - \-
      - \-
      - No
    * - WLKernel
      - Sim
      - PDB
      - SMILES
      - \-
      - \-
      - Yes

Input format
------------

DataSAIL has been designed to split biochemical datasets. Therefore, it accepts three different biochemical input
entities, namely proteins, molecules, and genomic input (either RNA, DNA, or whole genomes).

There are two scenarios of splitting, either interaction data into a cold-double split or a set of entities into a
single-cold split. In both cases, the entities must be represented either as sequences or as structures. How the
concrete input has to look like is described in the following.

Data Input
  - FASTA:
    A simple fasta file with sequence headers and sequences. An exception is FASTA input to MASH for genomic data.
    Here, the input has to be a folder of fasta files where each fasta file represents one genome.
  - PDB:
    This has to be a folder with PDB files. All files not ending with `.pdb` will be ignored.
  - SMILES:
    A TSV file with the molecule's ID in the first column and a SMILES string in the second column. Further columns
    will be ignored.
  - file:
    Either a numpy array as a pickle file or a similarity/distance matrix in TSV format. In case of the TSV file, the
    matrix has to be labeled with identifiers in both, a header row and the first column. If it is a pickle file, the
    order has to be given as additional argument.

To now split the data, DataSAIL needs to get the data in one of the formats described above. In case of interaction
data, both interacting entities need to be stored in either of these formats. In case of interaction data, you
additionally have to provide the interactions between both entities as TSV file with a header and one interaction per
row given by the two interacting IDs. Further columns will be ignored.
