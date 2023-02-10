import os
from typing import Generator, Tuple, Dict

import numpy as np

from datasail.reader.utils import read_csv, count_inter, read_similarity_file, ParseInfo


def read_protein_data(data, weights, sim, dist, max_sim, max_dist, inter, index) -> ParseInfo:
    if data.endswith(".fasta") or data.endswith(".fa"):
        proteins = parse_fasta(data)
    elif os.path.isfile(data):
        proteins = dict(read_csv(data, False, "\t"))
    elif os.path.isdir(data):
        proteins = dict(read_pdb_folder(data))
    else:
        raise ValueError()

    # parse the protein weights
    if weights is not None:
        protein_weights = dict((n, float(w)) for n, w in read_csv(weights, False, "\t"))
    elif inter is not None:
        protein_weights = dict(count_inter(inter, index))
    else:
        protein_weights = dict((p, 1) for p in list(proteins.keys()))

    # parse the protein similarity measure
    protein_similarity, protein_distance = None, None
    if sim is None and dist is None:
        protein_similarity = np.ones((len(proteins), len(proteins)))
        protein_names = list(proteins.keys())
        protein_threshold = 1
    elif sim is not None and os.path.isfile(sim):
        protein_names, protein_similarity = read_similarity_file(sim)
        protein_threshold = max_sim
    elif dist is not None and os.path.isfile(dist):
        protein_names, protein_distance = read_similarity_file(dist)
        protein_threshold = max_dist
    else:
        if sim is not None:
            protein_similarity = sim
            protein_threshold = max_sim
        else:
            protein_distance = dist
            protein_threshold = max_dist
        protein_names = list(proteins.keys())

    return protein_names, proteins, protein_weights, protein_similarity, protein_distance, protein_threshold


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
