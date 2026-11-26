"""Frozen pre-optimization matrix readers from 37ca771; tests only."""
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np

from datasail.reader.utils import MATRIX_INPUT


def read_clustering_file(filepath: Path, sep: str = "\t") -> Tuple[List[str], np.ndarray]:
    """
    Read a similarity or distance matrix from a file.

    Args:
        filepath: Path to the file storing the matrix in CSV format
        sep: Separator used to separate the values of the matrix

    Returns:
        A list of names of the entities and their pairwise interactions in and numpy array
    """
    names = []
    measures = []
    with open(filepath, "r") as data:
        for line in data.readlines()[1:]:
            parts = line.strip().split(sep)
            names.append(parts[0])
            measures.append([float(x) for x in parts[1:]])
    return names, np.array(measures)


def read_matrix_input(
        in_data: MATRIX_INPUT,
) -> Tuple[List[str], Union[np.ndarray, str]]:
    """
    Read the data from different types of similarity or distance.

    Args:
        in_data: Matrix data encoding the similarities/distances and the names of the samples

    Returns:
        Tuple of names of the data samples and a matrix holding their similarities/distances or a string encoding a
        method to compute the fore-mentioned
    """
    if isinstance(in_data, str):
        in_data = Path(in_data)
    if isinstance(in_data, Path) and in_data.is_file():
        names, similarity = read_clustering_file(in_data)
    elif isinstance(in_data, tuple):
        names, similarity = in_data
    elif isinstance(in_data, Callable):
        names, similarity = in_data()
    else:
        raise ValueError()
    return names, similarity
