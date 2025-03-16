import pickle
from argparse import Namespace
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Generator, Tuple, List, Optional, Dict, Union, Any, Callable, Iterable, Set
from collections.abc import Iterable

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem

from datasail.reader.validate import validate_user_args
from datasail.settings import get_default, SIM_ALGOS, DIST_ALGOS, UNK_LOCATION, format2ending, FASTA_FORMATS

DATA_INPUT = Optional[Union[str, Path, Dict[str, Union[str, np.ndarray]],
    Callable[..., Dict[str, Union[str, np.ndarray]]], Generator[Tuple[str, Union[str, np.ndarray]], None, None]]]
MATRIX_INPUT = Optional[Union[str, Path, Tuple[List[str], np.ndarray], Callable[..., Tuple[List[str], np.ndarray]]]]
DictMap = Dict[str, List[Dict[str, str]]]


@dataclass
class DataSet:
    type: Optional[str] = None
    format: Optional[str] = None
    args: Optional[Namespace] = None
    names: Optional[List[str]] = None
    id_map: Optional[Dict[str, str]] = None
    cluster_names: Optional[List[str]] = None
    num_clusters: int = 50
    data: Optional[Dict[str, Union[str, np.ndarray]]] = None
    cluster_map: Optional[Dict[str, str]] = None
    location: Optional[Path] = None
    weights: Optional[Dict[str, float]] = None
    cluster_weights: Optional[Dict[str, float]] = None
    classes: Optional[Dict[Any, int]] = None
    class_oh: Optional[np.ndarray] = None
    stratification: Optional[Dict[str, Any]] = None
    cluster_stratification: Optional[Dict[str, np.ndarray]] = None
    similarity: Optional[Union[np.ndarray, str]] = None
    cluster_similarity: Optional[Union[np.ndarray, str]] = None
    distance: Optional[Union[np.ndarray, str]] = None
    cluster_distance: Optional[Union[np.ndarray, str]] = None

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
                hv = 0
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
        if self.location is None or self.location == UNK_LOCATION:
            return "unknown"
        if isinstance(self.location, Path):
            if self.location.is_file():
                return self.location.stem
            return self.location.name
        return str(self.location)

    def get_location_path(self) -> Path:
        """
        Get the location of the dataset.

        Returns:
            The location of the dataset
        """
        if self.location is None or self.location == UNK_LOCATION or not self.location.exists():
            return Path("unknown." + format2ending(self.format))
        return self.location

    def strat2oh(self, name: Optional[str] = None, classes: Optional[Union[str, Set[str]]] = None) -> Optional[np.ndarray]:
        """
        Convert the stratification to a one-hot encoding.

        Args:
            name: Name of the sample to get the onehot encoding for
            class_: Class to get the onehot encoding for

        Returns:
            A one-hot encoding of the stratification
        """
        if classes is None:
            if name is None:
                raise ValueError("Either name or class must be provided.")
            classes = self.stratification[name]
        if not isinstance(classes, Iterable):
            classes = [classes]
        if self.classes is not None:
            return self.class_oh[[self.classes[class_] for class_ in classes]].sum(axis=0)
        return None

    def shuffle(self) -> None:
        """
        Shuffle this dataset randomly to introduce variance in the solution space.
        """
        if self.type is None:
            return

        self.names, self.similarity, self.distance = permute(self.names, self.similarity, self.distance)

        if self.cluster_names is not None:
            self.cluster_names, self.cluster_similarity, self.cluster_distance = \
                permute(self.cluster_names, self.cluster_similarity, self.cluster_distance)


def permute(names, similarity=None, distance=None) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Permute the order of the data the names list and the according distance or similarity matrix.

    Args:
        names: List of names of samples in the dataset
        similarity: Similarity matrix of datapoints in the dataset
        distance: Distance matrix of datapoints in the dataset

    Returns:
        Permuted names, similarity and distance matrix
    """
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


def read_csv(filepath: Path, sep: str = ",") -> Generator[Tuple[str, str], None, None]:
    """
    Read in a CSV file as pairs of data.

    Args:
        filepath: Path to the CSV file to read 2-tuples from
        sep: Separator used to separate the values in the CSV file

    Yields:
        Pairs of strings from the file
    """
    df = pd.read_csv(filepath, sep=sep)
    for index in df.index:
        yield df.iloc[index, :2]


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


def read_data(
        weights: DATA_INPUT,
        strats: DATA_INPUT,
        sim: MATRIX_INPUT,
        dist: MATRIX_INPUT,
        inter: Optional[List[Tuple[str, str]]],
        index: Optional[int],
        num_clusters: int,
        tool_args: str,
        dataset: DataSet,
) -> DataSet:
    """
    Compute the weight and distances or similarities of every entity.

    Args:
        weights: Weight file for the data
        strats: Stratification for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file
        num_clusters: Number of clusters to compute
        tool_args: Additional arguments for the tool
        dataset: A dataset object storing information on the read

    Returns:
        A dataset storing all information on that datatype
    """
    # parse the weights
    if isinstance(weights, Path) and weights.is_file():
        if weights.suffix[1:].lower() == "csv":
            dataset.weights = dict((n, float(w)) for n, w in read_csv(weights, ","))
        elif weights.suffix[1:].lower() == "tsv":
            dataset.weights = dict((n, float(w)) for n, w in read_csv(weights, "\t"))
        else:
            raise ValueError()
    elif isinstance(weights, dict):
        dataset.weights = weights
    elif isinstance(weights, Callable):
        dataset.weights = weights()
    elif isinstance(weights, Generator):
        dataset.weights = dict(weights)
    elif inter is not None:
        dataset.weights = {k: 0 for k in dataset.data.keys()}
        dataset.weights.update(dict(count_inter(inter, index)))
    else:
        dataset.weights = {k: 1 for k in dataset.data.keys()}

    # parse the stratification
    if isinstance(strats, Path) and strats.is_file():
        if strats.suffix[1:].lower() == "csv":
            dataset.stratification = dict(read_csv(strats, ","))
        elif strats.suffix[1:].lower() == "tsv":
            dataset.stratification = dict(read_csv(strats, "\t"))
        else:
            raise ValueError()
    elif isinstance(strats, dict):
        dataset.stratification = strats
    elif isinstance(strats, Callable):
        dataset.stratification = strats()
    elif isinstance(strats, Generator):
        dataset.stratification = dict(strats)
    else:
        dataset.stratification = {k: 0 for k in dataset.data.keys()}

    # .classes maps the individual classes to their index in one-hot encoding, important for non-numeric classes
    tmp_classes = set()
    for value in dataset.stratification.values():
        if isinstance(value, set):
            tmp_classes.update(value)
        else:
            tmp_classes.add(value)
    dataset.classes = {s: i for i, s in enumerate(tmp_classes)}
    dataset.class_oh = np.eye(len(dataset.classes))
    dataset.num_clusters = num_clusters

    # parse the similarity or distance measure
    if sim is None and dist is None:
        dataset.similarity, dataset.distance = get_default(dataset.type, dataset.format)
        dataset.names = list(dataset.data.keys())
    elif sim is not None and not (isinstance(sim, str) and sim.lower() in SIM_ALGOS):
        dataset.names, dataset.similarity = read_matrix_input(sim)
    elif dist is not None and not (isinstance(dist, str) and dist.lower() in DIST_ALGOS):
        dataset.names, dataset.distance = read_matrix_input(dist)
    else:
        if sim is not None:
            dataset.similarity = sim
        else:
            dataset.distance = dist
        dataset.names = list(dataset.data.keys())

    dataset.args = validate_user_args(dataset.type, dataset.format, sim, dist, tool_args)

    return dataset


def read_folder(folder_path: Path, file_extension: Optional[str] = None) -> Generator[Tuple[str, str], None, None]:
    """
    Read in all PDB file from a folder and ignore non-PDB files.

    Args:
        folder_path: Path to the folder storing the PDB files
        file_extension: File extension to parse, None if the files shall not be filtered

    Yields:
        Pairs of the PDB files name and the path to the file
    """
    for filename in folder_path.iterdir():
        if file_extension is None or filename.suffix[1:].lower() == file_extension.lower():
            yield filename.stem, filename


def read_data_input(data: DATA_INPUT, dataset: DataSet, read_dir: Callable[[DataSet, Path], None]) -> None:
    """
    Read in the data from different sources and store it in the dataset.

    Args:
        data: Data input
        dataset: Dataset to store the data in
        read_dir: Function to read in a directory
    """
    if isinstance(data, Path):
        if data.is_file():
            if data.suffix[1:] in FASTA_FORMATS:
                dataset.data = parse_fasta(data)
            elif data.suffix[1:].lower() == "tsv":
                dataset.data = dict(read_csv(data, sep="\t"))
            elif data.suffix[1:].lower() == "csv":
                dataset.data = dict(read_csv(data, sep=","))
            elif data.suffix[1:].lower() == "pkl":
                with open(data, "rb") as file:
                    dataset.data = dict(pickle.load(file))
            elif data.suffix[1:].lower() == "h5":
                with h5py.File(data) as file:
                    dataset.data = {k: np.array(file[k]) for k in file.keys()}
            elif data.suffix[1:].lower() == "sdf":
                dataset.data = read_sdf_file(data)
            else:
                raise ValueError("Unknown file format. Supported formats are: .fasta, .fna, .fa, tsv, .csv, .pkl, .h5")
        elif data.is_dir():
            read_dir(dataset, data)
        else:
            raise ValueError("Unknown data input type. Path encodes neither a file nor a directory.")
        dataset.location = data
    elif (isinstance(data, list) or isinstance(data, tuple)) and isinstance(data[0], Iterable) and len(data[0]) == 2:
        dataset.data = dict(data)
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError("Unknown data input type.")


def read_sdf_file(file: Path) -> Dict[str, str]:
    """
    Read in a SDF file and return the data as a dataset.

    Args:
        file: The file to read in

    Returns:
        The dataset containing the data
    """
    data = {}
    suppl = Chem.SDMolSupplier(str(file))
    for i, mol in enumerate(suppl):
        try:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{file.stem}_{i}"
            data[name] = Chem.MolToSmiles(mol)
        except:
            pass
    return data


def parse_fasta(path: Path = None) -> Dict[str, str]:
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
                entry_id = line[1:]
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map


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
