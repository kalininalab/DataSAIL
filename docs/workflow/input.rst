************
Input format
************

DataSAIL has been designed to split biochemical datasets. Therefore, it accepts three different biochemical input
entities, namely proteins, molecules, and genomic input (either RNA, DNA, or whole genomes).

There are two scenarios of splitting, either interaction data into a cold-double split or a set of entities into a
single-cold split. In both cases, the entities must be represented either as sequences or as structures. How the
concrete input has to look like is described in the following.

Data Input
  - FASTA:
    A simple fasta file with sequence headers and sequences. An exception is FASTA input to MASH for genomic data.
    Here, the input has to be a folder of fasta files where each fasta file represents one genome. The identifier in
    the FASTA files have to be free of whitespaces, otherwise, CD-HIT will have problem and might cause errors.
  - PDB:
    This has to be a folder with PDB files. All files not ending with :code:`.pdb` will be ignored.
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
