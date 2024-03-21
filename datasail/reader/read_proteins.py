from pathlib import Path
from typing import Tuple, Dict, List, Optional, Set

import numpy as np

from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.utils import DataSet, read_data, read_folder, DATA_INPUT, MATRIX_INPUT, read_data_input
from datasail.settings import P_TYPE, UNK_LOCATION, FORM_PDB, FORM_FASTA


def read_protein_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        strats: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
        num_clusters: Optional[int] = None,
        tool_args: str = "",
) -> DataSet:
    """
    Read in protein data, compute the weights, and distances or similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        strats: Stratification for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file
        num_clusters: Number of clusters to compute for this dataset
        tool_args: Additional arguments for the tool

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type=P_TYPE, location=UNK_LOCATION)

    def read_dir(ds: DataSet, path: Path) -> None:
        ds.data = dict(read_folder(path, "pdb"))

    read_data_input(data, dataset, read_dir)

    dataset.format = FORM_PDB if str(next(iter(dataset.data.values()))).endswith(".pdb") else FORM_FASTA

    dataset = read_data(weights, strats, sim, dist, inter, index, num_clusters, tool_args, dataset)
    dataset = remove_duplicate_values(dataset, dataset.data)

    return dataset


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


def extract_pdb_seqs(pdb_file: Path) -> Dict[str, str]:
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
