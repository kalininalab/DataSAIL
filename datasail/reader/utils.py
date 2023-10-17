import os
from argparse import Namespace
from dataclasses import dataclass, fields
from typing import Generator, Tuple, List, Optional, Dict, Union, Any, Callable

import numpy as np
import pandas as pd

from datasail.reader.validate import validate_user_args
from datasail.settings import get_default

DATA_INPUT = Optional[Union[str, Dict[str, str], Callable[..., Dict[str, str]], Generator[Tuple[str, str], None, None]]]
MATRIX_INPUT = Optional[Union[str, Tuple[List[str], np.ndarray], Callable[..., Tuple[List[str], np.ndarray]]]]
DictMap = Dict[str, List[Dict[str, str]]]


@dataclass
class DataSet:
    type: Optional[str] = None
    format: Optional[str] = None
    args: Optional[Union[Namespace, Tuple[Namespace, Namespace]]] = None
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
            elif isinstance(obj, np.ndarray):
                hv = 0  # hash(str(obj.data))
            elif isinstance(obj, Namespace):
                hv = hash(tuple(obj.__dict__.items()))
            else:
                hv = hash(obj)
            hash_val ^= hv
        return hash_val

    def __eq__(self, other: Any) -> bool:
        """
        Determine equality of two DataSets based on their hash value.

        Args:
            other: Other  object to compare to

        Returns:
            True if other object is a DataSet and contains the same information as this one.
        """
        return isinstance(other, DataSet) and hash(self) == hash(other)

    def get_name(self) -> str:
        """
        Compute the name of the dataset as the name of the file or the folder storing the data.

        Returns:
            Name of the dataset
        """
        if os.path.isfile(self.location):
            return os.path.splitext(os.path.basename(self.location))[0]
        elif os.path.isdir(self.location):
            return os.path.split(self.location)[-1]
        return self.location

    def shuffle(self):
        """
        Shuffle this dataset randomly to introduce variance in the solution space.
        """
        if self.type is None:
            return

        self.names, self.similarity, self.distance = permute(self.names, self.similarity, self.distance)

        if self.cluster_names is not None:
            self.cluster_names, self.cluster_similarity, self.cluster_distance = \
                permute(self.cluster_names, self.cluster_similarity, self.cluster_distance)


def permute(names, similarity=None, distance=None):
    permutation = np.random.permutation(len(names))
    names = [names[x] for x in permutation]
    if isinstance(similarity, np.ndarray):
        similarity = similarity[permutation, :]
        similarity = similarity[:, permutation]
    if isinstance(distance, np.ndarray):
        distance = distance[permutation, :]
        distance = distance[:, permutation]
    return names, similarity, distance


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
    df = pd.read_csv(filepath, sep="\t")
    for index in df.index:
        yield df.iloc[index, :2]


def read_matrix_input(
        in_data: MATRIX_INPUT, max_val: float = 1.0, default_names: Optional[List[str]] = None
) -> Tuple[List[str], Union[np.ndarray, str], float]:
    """
    Read the data from different types of similarity or distance.

    Args:
        in_data: Matrix data encoding the similarities/distances and the names of the samples
        max_val: Maximal value of the used metric, either distance or similarity
        default_names: Names to use as default, if max_val specifies a clustering method

    Returns:
        Tuple of names of the data samples, a matrix holding their similarities/distances or a string encoding a method
        to compute the fore-mentioned, and the threshold to apply when splitting
    """
    if isinstance(in_data, str):
        if os.path.isfile(in_data):
            names, similarity = read_clustering_file(in_data)
            threshold = max_val
        else:
            names = default_names
            similarity = in_data
            threshold = max_val
    elif isinstance(in_data, tuple):
        names, similarity = in_data
        threshold = max_val
    elif isinstance(in_data, Callable):
        names, similarity = in_data()
        threshold = max_val
    else:
        raise ValueError()
    return names, similarity, threshold


def read_data(
        weights: DATA_INPUT,
        sim: MATRIX_INPUT,
        dist: MATRIX_INPUT,
        max_sim: float,
        max_dist: float,
        inter: Optional[List[Tuple[str, str]]],
        index: Optional[int],
        tool_args: str,
        dataset: DataSet,
) -> DataSet:
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
        tool_args: Additional arguments for the tool
        dataset: A dataset object storing information on the read

    Returns:
        A dataset storing all information on that datatype
    """
    # parse the protein weights
    if weights is not None:
        if isinstance(weights, str):
            dataset.weights = dict((n, float(w)) for n, w in read_csv(weights))
        elif isinstance(weights, dict):
            dataset.weights = weights
        elif isinstance(weights, Callable):
            dataset.weights = weights()
        elif isinstance(weights, Generator):
            dataset.weights = dict(weights)
    elif inter is not None:
        dataset.weights = dict(count_inter(inter, index))
    else:
        dataset.weights = dict((p, 1) for p in list(dataset.data.keys()))

    # parse the protein similarity measure
    if sim is None and dist is None:
        dataset.similarity, dataset.distance = get_default(dataset.type, dataset.format)
        dataset.names = list(dataset.data.keys())
        dataset.threshold = 1
    elif sim is not None:
        dataset.names, dataset.similarity, dataset.threshold = \
            read_matrix_input(sim, max_sim, list(dataset.data.keys()))
    elif dist is not None:
        dataset.names, dataset.distance, dataset.threshold = \
            read_matrix_input(dist, max_dist, list(dataset.data.keys()))
    else:
        if sim is not None:
            dataset.similarity = sim
            dataset.threshold = max_sim
        else:
            dataset.distance = dist
            dataset.threshold = max_dist
        dataset.names = list(dataset.data.keys())

    dataset.args = validate_user_args(dataset.type, dataset.format, sim, dist, tool_args)

    return dataset


def read_folder(folder_path: str, file_extension: Optional[str] = None) -> Generator[Tuple[str, str], None, None]:
    """
    Read in all PDB file from a folder and ignore non-PDB files.

    Args:
        folder_path: Path to the folder storing the PDB files
        file_extension: File extension to parse, None if the files shall not be filtered

    Yields:
        Pairs of the PDB files name and the path to the file
    """
    for filename in os.listdir(folder_path):
        if file_extension is None or filename.endswith(file_extension):
            yield ".".join(filename.split(".")[:-1]), os.path.abspath(os.path.join(folder_path, filename))


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
