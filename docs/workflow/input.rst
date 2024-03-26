#############
Input formats
#############

DataSAIL has been designed to split biochemical datasets but can be applied to any other type of data if the user can
provide necessary information. Therefore, DataSAIL accepts different input formats as they are required by different
types of data.

CSV and TSV Files
#################

.. _files-xsv-label:

The standard way to share data in an effective way are :code:`.csv` and :code:`.tsv` files. In DataSAIL, these formats
are used to, e. g., transport data about molecules, weights of samples, or stratification. From these files, DataSAIL
only reads the first two columns. The first column has to contain the names of the samples and the second row the
according information (SMILES or FASTA string, weighting, stratification, ...). Also, the first row must be column
names, therefore, DataSAIL ignores the first row. Examples are given in :code:`tests/data/pipline/drug.tsv` and
:code:`tests/data/pipeline/drugs_weights.tsv`.

But they are also used to ship similarity and distance matrices. An
example is given in :code:`tests/data/pipeline/drug_sim.csv` and :code:`tests/data/pipeline/drug_dist.csv`. Here, the
first row and column contain the names of the samples and the rest of the matrix the similarities or distances between
the samples.

CSV and TSV files can also be used to transport interactions. An example is given in
:code:`tests/data/pipeline/inter.tsv`. Again, only the first two columns matter which specify which sample from the
e-entity with which sample from the f-entity interacts.

FASTA files
###########

.. _files-fasta-label:

FASTA files are widely used for various biological inputs. DataSAIL recognizes all files that end with :code:`.fa`,
:code:`.fna`, and :code:`.fasta` as FASTA files. In DataSAIL they are used to transport information about protein
sequences, nucleotide sequences (e.g. DNA or RNA), and whole genomes.

For Protein and Nucleotide Sequences
====================================

Sequence-based datasets are stored inside a single files. Each sequences must be identified with its name in a line
starting with a :code:`>`. All following lines are concatenated to form the sequence until there is an empty line, the
end of the file, or a line that starts with :code:`>` starting the next line. An example with protein sequences is
given in :code:`tests/data/pipline/seqs.fasta`.

For whole Genomes
=================

Genome input through FASTA files is a bit different to the format above. Here, each file contains all contigs, or reads
of one sample and the dataset is represented by a folder. Examples are given in :code:`tests/data/genomes`.

Pickle Files
############

.. _files-pickle-label:

From version 1.0.0 on, DataSAIL can also take embeddings as input. Here, the pickle file has to contain a dictionary
mapping the sample names to the embeddings. An example storing Morgan fingerprints of the molecules in
:code:`tests/data/pipeline/drugs.tsv` in a pickle file is given in :code:`tests/data/pipeline/morgan.pkl`.

HDF5 Files
##########

.. _files-hdf5-label:

Also, from version 1.0.0 on, DataSAIL supports the :code:`.h5` format. This format is used to store large datasets in
runtime and memory efficient way. Similar to Pickle files, the HDF5 file has to contain a dictionary mapping the sample
names to the embeddings. An example storing Morgan fingerprints of the molecules in
:code:`tests/data/pipeline/drugs.tsv` in a HDF5 file is given in :code:`tests/data/pipeline/morgan.h5`. To open and
convert it to a dictionary, the following code can be used:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('tests/data/pipeline/morgan.h5', 'r') as f:
        morgan = {k: np.array(v) for k, v in f.items()}

Example code for creation and reading of Pickle and HDF5 files can be found in :code:`tests/data/pipeline/embed.py`.

Molecular Input Files
#####################

.. _files-mol-label:

Molecules can be input as SMILES strings in TSV and CSV format as described above, but also using dedicated
fileformats. DataSAIL supports the following fileformats: :code:`.mol`, :code:`.mol2`, :code:`.mrv`, :code:`.pdb`,
:code:`.sdf`, :code:`.tpl`, and :code:`.xyz`. Files may only contain a single molecule (or molecular conformation),
except for :code:`.sdf` files, which can contain multiple molecules. The molecules are named based on their property
:code:`_Name` or their filename if the property is not set. In case of :code:`.sdf` files and molecules without
:code:`_Name` property, the index at which they are stored in the file is used as suffix to distinguish between
molecules in the same file.
