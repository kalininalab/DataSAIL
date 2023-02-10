from typing import Generator, Tuple, List, Optional, Dict, Union

import numpy as np

ParseInfo = Tuple[
    Optional[List[str]],
    Optional[Dict[str, str]],
    Optional[Dict[str, float]],
    Optional[Union[np.ndarray, str]],
    Optional[Union[np.ndarray, str]],
    float,
]


def count_inter(inter: List[Tuple[str, str]], mode: int) -> Generator[Tuple[str, int], None, None]:
    """
    Count interactions per protein or drug in a set of interactions.

    Args:
        inter: List of pairwise interactions of proteins and drugs
        mode: mode to read data for, either >protein> or >drug<

    Yields:
        Pairs of protein or drug names and the number of interactions they participate in
    """
    tmp = list(zip(*inter))
    keys = set(tmp[mode])
    for key in keys:
        yield key, tmp[mode].count(key)


def read_similarity_file(filepath: str, sep: str = "\t") -> Tuple[List[str], np.ndarray]:
    """
    Read a similarity or distance matrix from a file.

    Args:
        filepath: Path to the file storing the matrix in CSV format
        sep: separator used to separate the values of the matrix

    Returns:
        A list of names of the entities and their pairwise interactions in and numpy array
    """
    names = []
    similarities = []
    with open(filepath, "r") as data:
        for line in data.readlines()[1:]:
            parts = line.strip().split(sep)
            names.append(parts[0])
            similarities.append([float(x) for x in parts[1:]])
    return names, np.array(similarities)


def read_csv(filepath: str, header: bool = False, sep: str = "\t") -> Generator[Tuple[str, str], None, None]:
    """
    Read in a CSV file as pairs of data.

    Args:
        filepath: Path to the CSV file to read 2-tuples from
        header: bool flag indicating whether the file has a header-line
        sep: separator character used to separate the values

    Yields:
        Pairs of strings from the file
    """
    with open(filepath, "r") as inter:
        for line in inter.readlines()[(1 if header else 0):]:
            output = line.strip().split(sep)
            if len(output) >= 2:
                yield output[:2]
            else:
                yield output[0], output[0]
