import os
from typing import Generator, Tuple, Dict, List, Any, Optional, Set, Callable

import numpy as np

from datasail.reader.utils import read_csv, DataSet, read_data, read_folder, DATA_INPUT, MATRIX_INPUT


def read_protein_data(
        data: DATA_INPUT,
        weights: DATA_INPUT,
        sim: MATRIX_INPUT,
        dist: MATRIX_INPUT,
        max_sim: float,
        max_dist: float,
        id_map: Optional[str],
        inter: Optional[List[Tuple[str, str]]],
        index: Optional[int],
) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Read in protein data, compute the weights, and distances or similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        max_sim: Maximal similarity between entities in two splits
        max_dist: Maximal similarity between entities in one split
        id_map: Mapping of ids in case of duplicates in the dataset
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type="P", location=None)
    if isinstance(data, str):
        if data.split(".")[-1].lower() in {"fasta", "fa", "fna"}:
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
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError()

    dataset.format = "PDB" if os.path.exists(next(iter(dataset.data.values()))) else "FASTA"

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def remove_protein_duplicates(prefix: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Remove duplicates in protein input. This is done for FASTA input as well as for PDB input.

    Args:
       prefix: Prefix of the data. This is either 'e_' or 'f_'
       output_dir: Directory to store data to in case of detected duplicates
        **kwargs: Arguments for this data input

    Returns:
        Update arguments as teh location of the data might change and an ID-Map file might be added.
    """
    # read the data
    output_args = {prefix + k: v for k, v in kwargs.items()}
    if not isinstance(kwargs["data"], str):
        return output_args
    if kwargs["data"].split(".")[-1].lower() in {"fasta", "fa", "fna"}:
        sequences = parse_fasta(kwargs["data"])
    elif os.path.isfile(kwargs["data"]):
        sequences = dict(read_csv(kwargs["data"]))
    else:
        # input is PDB data. TODO: Identity detection with PDB files
        return output_args

    id_list = []  # unique ids
    id_map = {}  # mapping of all ids to their representative
    duplicate_found = False
    for idx, seq in sequences.items():
        for q_id in id_list:
            if seq == sequences[q_id]:
                id_map[idx] = q_id
                duplicate_found = True
        if idx not in id_map:
            id_list.append(idx)
            id_map[idx] = idx

    # no duplicates found, no further action necessary
    if not duplicate_found:
        return output_args

    # store the new FASTA file
    fasta_filename = os.path.abspath(os.path.join(output_dir, "tmp", prefix + "seqs.fasta"))
    with open(fasta_filename, "w") as out:
        for idx in id_list:
            print(f">{idx}\n{sequences[idx]}", file=out)
    output_args[prefix + "data"] = fasta_filename

    # store the mapping of IDs
    id_map_filename = os.path.join(output_dir, "tmp", prefix + "id_map.tsv")
    with open(id_map_filename, "w") as out:
        print("Name\tRepresentative", file=out)
        for idx, rep_id in id_map.items():
            print(idx, rep_id, sep="\t", file=out)
    output_args[prefix + "id_map"] = id_map_filename

    return output_args


def parse_fasta(path: str = None) -> Dict[str, str]:
    """
    Parse a FASTA file and do some validity checks if requested.

    Args:
        path: Path to the FASTA file

    Returns:
        Dictionary mapping sequences IDs to amino acid sequences
    """
    seq_map = {}

    with open(path, "r") as fasta:
        for line in fasta.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '>':
                entry_id = line[1:].replace(" ", "_")
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map


def check_pdb_pair(pdb_seqs1: List[str], pdb_seqs2: List[str]) -> bool:
    """
    Entry point for the comparison of two PDB files.

    Args:
        pdb_seqs1: filepath to the first PDB file
        pdb_seqs2: filepath to the second PDB file

    Returns:
        A boolean flag indicating (dis-)similarity of the two PDB files.
    """
    if len(pdb_seqs1) != len(pdb_seqs2):
        # If the number of sequence does not match, the PDB files cannot describe the same protein
        return False
    return check_pdb_pair_rec(pdb_seqs1, pdb_seqs2, 0, set(), np.full((len(pdb_seqs1), len(pdb_seqs1)), -1))


def check_pdb_pair_rec(
        pdb_seqs1: List[str],
        pdb_seqs2: List[str],
        index1: int,
        blocked: Set[int],
        dp_table: np.ndarray
) -> bool:
    """
    Check if two pdb files contain the same protein. This is done in recursive manner by finding a match for one
    sequence in the first pdb file and check recursively if all sequences can be matched based on that assignment.
    This works somewhat like DFS on a tree of all possible assignments.

    Args:
        pdb_seqs1: List of amino acid sequences from the first PDB file.
        pdb_seqs2: List of amino acid sequences from the second PDB file.
        index1: The index of the sequence in pdb_seqs1 looking for a mate in pdb_seqs2.
        blocked: List of indices already assigned in higher iteration of the recursion
        dp_table: Table of already computed sequence similarities

    Returns:
        True if the two lists of amino acid sequences from the two files can be matched, otherwise False
    """
    if index1 == len(pdb_seqs1):  # every seq in pdb_seqs1 has found a mate in pdb_seqs2
        return True

    # iterate over all sequences in pdb_seqs2 ...
    for index in range(len(pdb_seqs2)):
        if index in blocked:
            continue

        # ... and check if they match the current sequence from pdb_seqs1
        if dp_table[index1, index] == -1:
            dp_table[index1, index] = seqs_equality(pdb_seqs1[index1], pdb_seqs2[index])

        # if I found a match, go deeper recursively and check if I can match the rest as well
        if dp_table[index1, index] == 1:
            blocked.add(index)
            if check_pdb_pair_rec(pdb_seqs1, pdb_seqs2, index + 1, blocked, dp_table):
                return True
            blocked.remove(index)
    return False


def seqs_equality(seq1: str, seq2: str) -> float:
    """
    Compute if two sequences are similar or not.

    Args:
        seq1: First sequence to compare
        seq2: Second sequence to compare

    Returns:
        A similarity measure for the two sequences
    """
    return 1.0 if seq1 == seq2 else 0.0


def extract_pdb_seqs(pdb_file: str) -> Dict[str, str]:
    """
    Extract all amino acid sequences from a PDB file.

    Args:
        pdb_file: filepath to the PDB file in question.

    Returns:
        A dictionary of all chain ids mapping to their amino acid sequence.
    """
    seqs = {}
    with open(pdb_file, "r") as pdb:
        for line in pdb.readlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = line[20:22].strip()
                if res_num not in seqs:
                    seqs[res_num] = ""
                seqs[res_num] += line[17:20].strip()
    return seqs
