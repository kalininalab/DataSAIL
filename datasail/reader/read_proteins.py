import os
from typing import Generator, Tuple, Dict

import numpy as np

from datasail.reader.utils import read_csv, count_inter, read_similarity_file, DataSet


def read_protein_data(data, weights, sim, dist, max_sim, max_dist, inter, index) -> DataSet:
    """
    Parse protein data into a dataset.

    Args:
        data: Location where the actual protein data is stored
        weights: weights of the proteins in the entity
        sim: similarity metric between pairs of proteins
        dist: distance metrix between pairs of proteins
        max_sim: maximal similarity of pairs of proteins when splitting
        max_dist: maximal distance of pairs of proteins when splitting
        inter: interaction of the proteins and another entity
        index: position of the proteins in the interactions, either 0 or 1

    Returns:
        molecules parsed into a dataset
    """
    dataset = DataSet(type="P")
    if data.endswith(".fasta") or data.endswith(".fa"):
        dataset.data = parse_fasta(data)
    elif os.path.isfile(data):
        dataset.data = dict(read_csv(data, False, "\t"))
        dataset.location = data
    elif os.path.isdir(data):
        dataset.data = dict(read_pdb_folder(data))
        dataset.location = data
    else:
        raise ValueError()

    # parse the protein weights
    if weights is not None:
        dataset.weights = dict((n, float(w)) for n, w in read_csv(weights, False, "\t"))
    elif inter is not None:
        dataset.weights = dict(count_inter(inter, index))
    else:
        dataset.weights = dict((p, 1) for p in list(dataset.data.keys()))

    # parse the protein similarity measure
    if sim is None and dist is None:
        dataset.similarity = np.ones((len(dataset.data), len(dataset.data)))
        dataset.names = list(dataset.data.keys())
        dataset.threshold = 1
    elif sim is not None and os.path.isfile(sim):
        dataset.names, dataset.similarity = read_similarity_file(sim)
        dataset.threshold = max_sim
    elif dist is not None and os.path.isfile(dist):
        dataset.names, dataset.distance = read_similarity_file(dist)
        dataset.threshold = max_dist
    else:
        if sim is not None:
            dataset.similarity = sim
            dataset.threshold = max_sim
        else:
            dataset.distance = dist
            dataset.threshold = max_dist
        dataset.names = list(dataset.data.keys())

    return dataset


def read_pdb_folder(folder_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read in all PDB file from a folder and ignore non-PDB files.

    Args:
        folder_path: Path to the folder storing the PDB files

    Yields:
        Pairs of the PDB files name and the path to the file
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdb"):
            yield filename[:-4], os.path.join(folder_path, filename)


def parse_fasta(path=None, left_split=None, right_split=' ', check_dups=False) -> Dict[str, str]:
    """
    Parse a FASTA file and do some validity checks if requested.

    Args:
        path:
        left_split:
        right_split:
        check_dups:

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
                if check_dups and entry_id in seq_map:
                    print(f'Duplicate entry in fasta input detected: {entry_id}')
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map
