import os
from dataclasses import dataclass, fields
from typing import Generator, Tuple, List, Optional, Dict, Union, Any, Callable

import numpy as np


@dataclass
class DataSet:
    type: Optional[str] = None
    format: Optional[str] = None
    args: str = ""
    names: Optional[List[str]] = None
    id_map: Optional[Dict[str, str]] = None
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

    def __hash__(self) -> int:
        """
        Compute the hash value for this dataset to be used in caching. Therefore, the hash is computed on properties
        that do not change during clustering.

        Returns:
            The cluster-insensitive hash-value of the instance.
        """
        hash_val = 0
        for field in filter(lambda f: "cluster" not in f.name, fields(DataSet)):
            obj = getattr(self, field.name)
            if obj is None:
                hv = 0
            elif isinstance(obj, dict):
                hv = hash(tuple(obj.items()))
            elif isinstance(obj, list):
                hv = hash(tuple(obj))
            # elif isinstance(obj, np.ndarray):
            #     hv = hash(str(obj.data))
            else:
                hv = hash(obj)
            hash_val ^= hv
        return hash_val

    def get_name(self) -> str:
        """
        Compute the name of the dataset as the name of the file or the folder storing the data.

        Returns:
            Name of the dataset
        """
        return ".".join(self.location.split(os.path.sep)[-1].split(".")[:-1]).replace("/", "_")


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


def read_data(
        weights: Optional[str],
        sim: str,
        dist: str,
        max_sim: float,
        max_dist: float,
        id_map: Optional[str],
        inter: Optional[List[Tuple[str, str]]],
        index: int,
        dataset: DataSet,
) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Compute the weight and distances or similarities of every entity.

    Args:
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        max_sim: Maximal similarity between entities in two splits
        max_dist: Maximal similarity between entities in one split
        id_map: Mapping of ids in case of duplicates in the dataset
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

    # parse mapping of duplicates
    if id_map is None:
        dataset.id_map = {k: k for k in dataset.names}
        print("No duplicates found.")
        return dataset, inter

    dataset.id_map = dict(read_csv(id_map))

    if inter is not None:
        new_inter = []
        for a, b in inter:
            if index == 0 and a in dataset.id_map:
                new_inter.append((dataset.id_map[a], b))
            elif index == 1 and b in dataset.id_map:
                new_inter.append((a, dataset.id_map[b]))
        inter = new_inter

    # update weights
    new_weights = dict()
    for name, weight in dataset.weights.items():
        if name not in dataset.id_map:
            continue
        new_name = dataset.id_map[name]
        if new_name not in new_weights:
            new_weights[new_name] = 0
        new_weights[new_name] += weight
    dataset.weights = new_weights

    return dataset, inter


def get_default(data_type: str, data_format: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the default clustering method for a specific type of data and a specific format.

    Args:
        data_type: Type of data as string representation
        data_format: Format encoded as string

    Returns:
        Tuple of the names of the method to use to compute either the similarity or distance for the input
    """
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


def get_prefix_args(prefix, **kwargs) -> Dict[str, Any]:
    """
    Remove prefix from keys and return those key-value-pairs.

    Args:
        prefix: Prefix to use for selecting key-value-pairs
        **kwargs: Keyword arguments provided to the program

    Returns:
        A subset of the key-value-pairs
    """
    return {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)}
