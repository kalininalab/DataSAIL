import os
from typing import Generator, Tuple, Dict

from datasail.reader.utils import read_csv, DataSet, read_data


def read_protein_data(data, weights, sim, dist, max_sim, max_dist, inter, index) -> DataSet:
    """
    Read in protein data, compute the weights, and distances or similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        max_sim: Maximal similarity between entities in two splits
        max_dist: Maximal similarity between entities in one split
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type="P")
    if data.endswith(".fasta") or data.endswith(".fa") or data.endswith(".fna"):
        dataset.data = parse_fasta(data)
        dataset.format = "FASTA"
    elif os.path.isfile(data):
        dataset.data = dict(read_csv(data))
        dataset.format = "FASTA"
    elif os.path.isdir(data):
        dataset.data = dict(read_folder(data, ".pdb"))
        dataset.format = "PDB"
    else:
        raise ValueError()
    dataset.location = data

    return read_data(weights, sim, dist, max_sim, max_dist, inter, index, dataset)


def read_folder(folder_path: str, file_extension: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read in all PDB file from a folder and ignore non-PDB files.

    Args:
        folder_path: Path to the folder storing the PDB files
        file_extension: File extension to parse

    Yields:
        Pairs of the PDB files name and the path to the file
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            yield ".".join(filename.split(".")[:-1]), os.path.abspath(os.path.join(folder_path, filename))


def parse_fasta(path=None, left_split=None, right_split=' ', check_duplicates=False) -> Dict[str, str]:
    """
    Parse a FASTA file and do some validity checks if requested.

    Args:
        path: Path to the FASTA file
        left_split: Char to use to split on the left side
        right_split: Char to use to split on the right side
        check_duplicates: Flag to check duplicates

    Returns:
        Dictionary mapping sequences IDs to amino acid sequences
    """
    seq_map = {}

    with open(path, "r") as fasta:
        for line in fasta.readlines():
            line = line.replace('\n', '')
            if len(line) == 0:
                continue
            if line[0] == '>':
                entry_id = line[1:].replace('β', 'beta')

                if entry_id[:3] == 'sp|' or entry_id[:3] == 'tr|':  # Detect uniprot/tremble ID strings
                    entry_id = entry_id.split('|')[1]

                if left_split is not None:
                    entry_id = entry_id.split(left_split, 1)[1]
                if right_split is not None:
                    entry_id = entry_id.split(right_split, 1)[0]
                if check_duplicates and entry_id in seq_map:
                    print(f'Duplicate entry in fasta input detected: {entry_id}')
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map
