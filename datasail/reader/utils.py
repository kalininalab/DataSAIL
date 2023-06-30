import os
from dataclasses import dataclass, fields
from typing import Generator, Tuple, List, Optional, Dict, Union, Any, Callable

import numpy as np

LIST_INPUT = Union[str, List[str], Callable[..., List[str]], Generator[str, None, None]]
DATA_INPUT = Optional[Union[str, Dict[str, str], Callable[..., Dict[str, str]], Generator[Tuple[str, str], None, None]]]
MATRIX_INPUT = Optional[Union[str, Tuple[List[str], np.ndarray], Callable[..., Tuple[List[str], np.ndarray]]]]
DictMap = Dict[str, List[Dict[str, str]]]


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
            elif isinstance(obj, np.ndarray):
                hv = 0  # hash(str(obj.data))
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
        permutation = np.random.permutation(len(self.names))
        self.names = [self.names[x] for x in permutation]
        if isinstance(self.similarity, np.ndarray):
            self.similarity = self.similarity[permutation, :]
            self.similarity = self.similarity[:, permutation]
        if isinstance(self.distance, np.ndarray):
            self.distance = self.distance[permutation, :]
            self.distance = self.distance[:, permutation]

        if self.cluster_names is not None:
            cluster_permutation = np.random.permutation(len(self.cluster_names))
            self.cluster_names = [self.cluster_names[x] for x in cluster_permutation]
            if isinstance(self.cluster_similarity, np.ndarray):
                self.cluster_similarity = self.cluster_similarity[cluster_permutation, :]
                self.cluster_similarity = self.cluster_similarity[:, cluster_permutation]
            if isinstance(self.cluster_distance, np.ndarray):
                self.cluster_distance = self.cluster_distance[cluster_permutation, :]
                self.cluster_distance = self.cluster_distance[:, cluster_permutation]


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
    match in_data:
        case x if isinstance(x, str):
            if os.path.isfile(in_data):
                names, similarity = read_clustering_file(in_data)
                threshold = max_val
            else:
                names = default_names
                similarity = in_data
                threshold = max_val
        case x if isinstance(x, tuple):
            names, similarity = in_data
            threshold = max_val
        case x if isinstance(x, Callable):
            names, similarity = in_data()
            threshold = max_val
        case _:
            raise ValueError()
    return names, similarity, threshold


def read_data(
        weights: DATA_INPUT,
        sim: MATRIX_INPUT,
        dist: MATRIX_INPUT,
        max_sim: float,
        max_dist: float,
        id_map: Optional[str],
        inter: Optional[List[Tuple[str, str]]],
        index: Optional[int],
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
        match weights:
            case str():
                dataset.weights = dict((n, float(w)) for n, w in read_csv(weights))
            case dict():
                dataset.weights = weights
            case x if isinstance(x, Callable):
                dataset.weights = weights()
            case x if isinstance(x, Generator):
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


def get_default(data_type: str, data_format: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the default clustering method for a specific type of data and a specific format.

    Args:
        data_type: Type of data as string representation
        data_format: Format encoded as string

    Returns:
        Tuple of the names of the method to use to compute either the similarity or distance for the input
    """
    match data_type:
        case "P":
            if data_format == "PDB":
                return "foldseek", None
            elif data_format == "FASTA":
                return "cdhit", None
        case _ if "M" and data_format == "SMILES":
            return "ecfp", None
        case _ if "G" and data_format == "FASTA":
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
