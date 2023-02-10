import os.path
from typing import Tuple, Generator, Dict, List, Optional, Union, Any

import numpy as np

from datasail.reader.read_genomes import read_genome_data
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_other import read_other_data
from datasail.reader.read_proteins import read_protein_data
from datasail.reader.utils import read_csv, ParseInfo


def read_data(**kwargs) -> Tuple[Tuple[str, ParseInfo], Tuple[str, ParseInfo], Optional[List[Tuple[str, str]]]]:
    """
    Read data from the input arguments.

    Args:
        **kwargs: Arguments from commandline

    Returns:
        Two tuples consisting of
          - The names of the current clusters
          - The mapping from cluster names to cluster representatives
          - Symmetric matrix of pairwise similarities between the current clusters
          - Symmetric matrix of pairwise similarities between the current clusters
          - Mapping from current clusters to their weights
        for both, protein data and drug data, as well as a list of interactions between
    """
    # TODO: Semantic checks of arguments
    inter = list(tuple(x) for x in read_csv(kwargs["inter"], False, "\t")) if kwargs["inter"] else None
    e_names, e_data, e_weights, e_similarity, e_distance, e_threshold = read_data_type(kwargs["e_type"])(
        kwargs["e_data"], kwargs["e_weights"], kwargs["e_sim"],
        kwargs["e_dist"], kwargs["e_max_sim"], kwargs["e_max_dist"], inter, 0
    )
    f_names, f_data, f_weights, f_similarity, f_distance, f_threshold = read_data_type(kwargs["f_type"])(
        kwargs["f_data"], kwargs["f_weights"], kwargs["f_sim"],
        kwargs["f_dist"], kwargs["f_max_sim"], kwargs["f_max_dist"], inter, 1
    )
    if kwargs["e_type"] is None:
        e_names, e_data, e_weights, e_similarity, e_distance, e_threshold = \
            f_names, f_data, f_weights, f_similarity, f_distance, f_threshold
        f_data, f_names, f_weights, f_similarity, f_distance, f_threshold = None, None, None, None, None, None
    return ((
        kwargs["e_type"], (
            e_names,
            e_data,
            e_weights,
            e_similarity,
            e_distance,
            e_threshold,
        )), (
        kwargs["f_type"], (
            f_names,
            f_data,
            f_weights,
            f_similarity,
            f_distance,
            f_threshold,
        )), inter,
    )


def read_data_type(data_type):
    if data_type == "P":
        return read_protein_data
    if data_type == "M":
        return read_molecule_data
    if data_type == "G":
        return read_genome_data
    if data_type == "O":
        return read_other_data
    return read_none_data


def read_none_data(*args, **kwargs):
    return None, None, None, None, None, None
