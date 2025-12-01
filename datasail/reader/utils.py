import hashlib
import pickle
from pathlib import Path
from typing import Generator, Literal, Optional, Union, Any, Callable, Iterable
from collections.abc import Iterable

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem

from datasail.dataset import DataSet
from datasail.validation.validate import validate_user_args
from datasail.constants import SRC_CL, SRC_ID, TEC_R, get_default, DATA_INPUT, MATRIX_INPUT, SIM_ALGOS, DIST_ALGOS, FASTA_FORMATS


class DimTechnique:
    def __init__(self, clustering: bool, dim: int):
        self.clustering = clustering
        self.dim = dim
        
    def __str__(self) -> str:
        return f"{SRC_CL if self.clustering else SRC_ID}{self.dim}"
    
    def __repr__(self) -> str:
        return self.__str__()


class Technique:
    def __init__(self, dims: list[DimTechnique] | Literal["R"]):
        if dims == TEC_R:
            self.dims = [DimTechnique(False, -1)]
        else:
            self.dims = dims
    
    def is_random(self) -> bool:
        return self.dims[0].dim == -1
    
    def is_oned(self) -> bool:
        return len(self.dims) == 1 and self.dims[0].dim != -1
    
    def is_clustered(self) -> bool:
        return any(d.clustering for d in self.dims)

    def __len__(self) -> int:
        return len(self.dims)

    def __getitem__(self, index: int) -> DimTechnique:
        return self.dims[index]
    
    def __str__(self) -> str:
        if self.is_random():
            return TEC_R
        return "-".join([str(d) for d in self.dims])
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __hash__(self) -> int:
        return self.__str__().__hash__()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Technique):
            return False
        return str(self) == str(other)
    
    @staticmethod
    def read_techniques(techniques: list[str]) -> list["Technique"]:
        output = []
        for request in techniques:
            if request == TEC_R:
                output.append(Technique(TEC_R))
                continue
            global_cluster = request[0] == SRC_CL
            tech_list = []
            for tech in request.split("-"):
                if tech[0].isdigit():
                    tech_list.append(DimTechnique(global_cluster, int(tech)))
                    continue
                if tech[0] not in {SRC_CL, SRC_ID}:
                    raise ValueError(f"Unknown clustering specifier {tech[0]} in technique {tech}.")
                tech_list.append(DimTechnique(tech[0] == SRC_CL, int(tech[1:])))
            output.append(Technique(tech_list))
        return output


def read_data(
        weights: DATA_INPUT,
        strats: DATA_INPUT,
        sim: MATRIX_INPUT,
        dist: MATRIX_INPUT,
        inter: Optional[list[tuple]],
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
    
    dataset.stratification = convert_stratification(dataset)
    num_classes = len(next(iter(dataset.stratification.values())))
    dataset.class_oh = np.eye(num_classes)
    dataset.classes = {s: i for i, s in enumerate(range(num_classes))}

    # # .classes maps the individual classes to their index in one-hot encoding, important for non-numeric classes
    # tmp_classes = set()
    # for value in dataset.stratification.values():
    #     if isinstance(value, set):
    #         tmp_classes.update(value)
    #     else:
    #         tmp_classes.add(value)
    # dataset.classes = {s: i for i, s in enumerate(tmp_classes)}
    # dataset.class_oh = np.eye(len(dataset.classes))
    # dataset.num_clusters = num_clusters

    dataset.args = validate_user_args(dataset.type, dataset.format, sim, dist, tool_args)

    return dataset


def convert_stratification(dataset: DataSet) -> dict[str, Any]:
    if dataset.stratification is None:
        stratification = {name: [0] for name in dataset.names}
    else:
        val = next(iter(dataset.stratification.values()))
        if isinstance(val, set):
            classes = list(sorted(set.union(*[set(x) for x in dataset.stratification.values()])))
            ohe = np.eye(len(classes))
            ohe_map = {c: ohe[i] for i, c in enumerate(classes)}
            stratification = {name: np.sum(np.stack([ohe_map[n] for n in dataset.stratification[name]]), axis=0) for name in dataset.names}
        elif not hasattr(val, "__getitem__") or isinstance(val, str):
            classes = list(set(dataset.stratification.values()))
            ohe = np.eye(len(classes))
            ohe_map = {c: ohe[i] for i, c in enumerate(classes)}
            stratification = {name: ohe_map[dataset.stratification[name]] for name in dataset.names}
        else:
            strat = pd.DataFrame(dataset.stratification).T
            for col in strat.columns:
                vals = strat[col].unique().size
                if vals > 2:
                    # convert to one-hot encoding
                    ohe = pd.get_dummies(strat[col], prefix=col)
                    strat = pd.concat([strat, ohe], axis=1)
                    strat = strat.astype({c: int for c in ohe.columns}, errors="ignore")
                if vals != 2:
                    strat.drop(columns=col, inplace=True, errors="ignore")
            stratification = strat.T.to_dict(orient="list")
    return stratification


def read_input_data(data: DATA_INPUT, dataset: DataSet, read_dir: Callable[[DataSet, Path], None]) -> None:
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
                dataset.data = read_fasta(data)
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


def read_folder(folder_path: Path, file_extension: Optional[str] = None) -> Generator[tuple, None, None]:
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


def read_matrix_input(in_data: MATRIX_INPUT) -> tuple[list[str], Union[np.ndarray, str]]:
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


def read_csv(filepath: Path, sep: str = ",", num_positions: int = 2) -> Generator[tuple, None, None]:
    """
    Read in a CSV file as pairs of data.

    Args:
        filepath: Path to the CSV file to read 2-tuples from
        sep: Separator used to separate the values in the CSV file
        num_positions: number of positions to read from each line

    Yields:
        num_positions-tuple of strings from the file
    """
    df = pd.read_csv(filepath, sep=sep)
    for index in df.index:
        yield df.iloc[index, :num_positions]


def read_sdf_file(file: Path) -> dict[str, str]:
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


def read_fasta(path: Path = None) -> dict[str, str]:
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


def read_clustering_file(filepath: Path, sep: str = "\t") -> tuple[list[str], np.ndarray]:
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


def count_inter(inter: list[tuple], mode: int) -> Generator[tuple[str, int], None, None]:
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


def get_prefix_args(prefix, **kwargs) -> dict[str, Any]:
    """
    Remove prefix from keys and return those key-value-pairs.

    Args:
        prefix: Prefix to use for selecting key-value-pairs
        **kwargs: Keyword arguments provided to the program

    Returns:
        A subset of the key-value-pairs
    """
    return {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)}
