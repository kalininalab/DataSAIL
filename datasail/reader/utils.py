import os
from dataclasses import dataclass, fields
from typing import Generator, Tuple, List, Optional, Dict, Union

import numpy as np


@dataclass
class DataSet:
    type: Optional[str] = None
    format: Optional[str] = None
    args: str = ""
    names: Optional[List[str]] = None
    cluster_names: Optional[List[str]] = None
    data: Optional[Dict[str, str]] = None
    cluster_map: Optional[Dict[str, str]] = None
    location: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    cluster_weights: Optional[Dict[str, float]] = None
    similarity: Optional[Union[np.ndarray, str]] = None
    cluster_similarity: Optional[Union[np.ndarray, str]] = None
    distance: Optional[Union[np.ndarray, str]] = None
    cluster_distance: Optional[Union[np.ndarray, str]] = None
    threshold: Optional[float] = None

    def __hash__(self):
        hash_val = 0
        for field in filter(lambda f: "cluster" not in f.name, fields(DataSet)):
            obj = getattr(self, field.name)
            if obj is None:
                hv = 0
            elif isinstance(obj, dict):
                hv = hash(tuple(obj.items()))
            elif isinstance(obj, list):
                hv = hash(tuple(obj))
            elif isinstance(obj, np.ndarray):
                hv = hash(str(obj.data))
            else:
                hv = hash(obj)
            hash_val ^= hv
        return hash_val


def count_inter(inter: List[Tuple[str, str]], mode: int) -> Generator[Tuple[str, int], None, None]:
    """
    Count interactions per entity in a set of interactions.

    Args:
        inter: List of pairwise interactions of entities
        mode: Position where to read the data from, first or second entity

    Yields:
        Pairs of entity name and the number of interactions they participate in
    """
    tmp = list(zip(*inter))
    keys = set(tmp[mode])
    for key in keys:
        yield key, tmp[mode].count(key)


def read_clustering_file(filepath: str, sep: str = "\t") -> Tuple[List[str], np.ndarray]:
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


def read_csv(filepath: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read in a CSV file as pairs of data.

    Args:
        filepath: Path to the CSV file to read 2-tuples from

    Yields:
        Pairs of strings from the file
    """
    with open(filepath, "r") as inter:
        for line in inter.readlines()[1:]:
            output = line.strip().split("\t")
            if len(output) >= 2:
                yield output[:2]
            else:
                yield output[0], output[0]


def read_data(weights, sim, dist, max_sim, max_dist, inter, index, dataset: DataSet) -> DataSet:
    """
    Compute the weight and distances or similarities of every entity.

    Args:
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        max_sim: Maximal similarity between entities in two splits
        max_dist: Maximal similarity between entities in one split
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file
        dataset: A dataset object storing information on the read

    Returns:
        A dataset storing all information on that datatype
    """
    # parse the protein weights
    if weights is not None:
        dataset.weights = dict((n, float(w)) for n, w in read_csv(weights))
    elif inter is not None:
        dataset.weights = dict(count_inter(inter, index))
    else:
        dataset.weights = dict((p, 1) for p in list(dataset.data.keys()))

    # parse the protein similarity measure
    if sim is None and dist is None:
        dataset.similarity, dataset.distance = get_default(dataset.type, dataset.format)
        dataset.names = list(dataset.data.keys())
        dataset.threshold = 1
    elif sim is not None and os.path.isfile(sim):
        dataset.names, dataset.similarity = read_clustering_file(sim)
        dataset.threshold = max_sim
    elif dist is not None and os.path.isfile(dist):
        dataset.names, dataset.distance = read_clustering_file(dist)
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


def get_default(data_type: str, data_format: str) -> Tuple[
    Optional[Union[np.ndarray, str]], Optional[Union[np.ndarray, str]]
]:
    if data_type == "P":
        if data_format == "PDB":
            return "foldseek", None
        elif data_format == "FASTA":
            return "cdhit", None
    elif data_type == "M":
        if data_format == "SMILES":
            return "ecfp", None
    elif data_type == "G":
        if data_format == "FASTA":
            return None, "mash"
    return None, None
