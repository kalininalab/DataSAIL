import os.path
from typing import Tuple, Generator, Dict, List, Optional, Union, Any

import numpy as np

from datasail.utils.utils import parse_fasta

ParseInfo = Tuple[
    Optional[List[str]],
    Optional[Dict[str, str]],
    Optional[Dict[str, float]],
    Optional[Union[np.ndarray, str]],
    Optional[Union[np.ndarray, str]],
    float
]


def read_data(**kwargs) -> Tuple[ParseInfo, ParseInfo, Optional[List[Tuple[str, str]]]]:
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

    # parse the proteins
    if kwargs["protein_data"] is not None:
        if kwargs["protein_data"].endswith(".fasta") or kwargs["protein_data"].endswith(".fa"):
            proteins = parse_fasta(kwargs["protein_data"])
        elif os.path.isfile(kwargs["protein_data"]):
            proteins = dict(read_csv(kwargs["protein_data"], False, "\t"))
        elif os.path.isdir(kwargs["protein_data"]):
            proteins = dict(read_pdb_folder(kwargs["protein_data"]))
        else:
            raise ValueError()

        # parse the protein weights
        if kwargs["protein_weights"] is not None:
            protein_weights = dict((n, float(w)) for n, w in read_csv(kwargs["protein_weights"], False, "\t"))
        elif inter is not None:
            protein_weights = dict(count_inter(inter, "prots"))
        else:
            protein_weights = dict((p, 1) for p in list(proteins.keys()))

        # parse the protein similarity measure
        protein_similarity, protein_distance = None, None
        if kwargs["protein_sim"] is None and kwargs["protein_dist"] is None:
            protein_similarity = np.ones((len(proteins), len(proteins)))
            protein_names = list(proteins.keys())
            protein_threshold = 1
        elif kwargs["protein_sim"] is not None and os.path.isfile(kwargs["protein_sim"]):
            protein_names, protein_similarity = read_similarity_file(kwargs["protein_sim"])
            protein_threshold = kwargs.get("protein_max_sim", 1)
        elif kwargs["protein_dist"] is not None and os.path.isfile(kwargs["protein_dist"]):
            protein_names, protein_distance = read_similarity_file(kwargs["protein_dist"])
            protein_threshold = kwargs.get("protein_max_dist", 1)
        else:
            if kwargs["protein_sim"] is not None:
                protein_similarity = kwargs["protein_sim"]
                protein_threshold = kwargs.get("protein_max_sim", 1)
            else:
                protein_distance = kwargs["protein_dist"]
                protein_threshold = kwargs.get("protein_max_dist", 1)
            protein_names = list(proteins.keys())
    else:
        proteins, protein_names, protein_weights, protein_similarity, protein_distance, protein_threshold = \
            None, None, None, None, None, 0

    # parse molecules
    if kwargs["ligand_data"] is not None:
        drugs = dict(read_csv(kwargs["ligand_data"], False, "\t"))

        # parse molecular weights
        if kwargs["ligand_weights"] is not None:
            drug_weights = dict((x, float(v)) for x, v in read_csv(kwargs["ligand_weights"], False, "\t"))
        elif inter is not None:
            drug_weights = dict(count_inter(inter, "drugs"))
        else:
            drug_weights = dict((d, 1) for d in list(drugs.keys()))

        # parse molecular similarity
        drug_similarity, drug_distance = None, None
        if kwargs["ligand_sim"] is None and kwargs["ligand_dist"] is None:
            drug_similarity = np.ones((len(drugs), len(drugs)))
            drug_names = list(drugs.keys())
            drug_threshold = 1
        elif kwargs["ligand_sim"] is not None and os.path.isfile(kwargs["ligand_sim"]):
            drug_names, drug_similarity = read_similarity_file(kwargs["ligand_sim"])
            drug_threshold = kwargs.get("ligand_max_sim", 1)
        elif kwargs["ligand_dist"] is not None and os.path.isfile(kwargs["ligand_dist"]):
            drug_names, drug_distance = read_similarity_file(kwargs["ligand_dist"])
            drug_threshold = kwargs.get("ligand_max_dist", 1)
        else:
            if kwargs["ligand_sim"] is not None:
                drug_similarity = kwargs["ligand_sim"]
                drug_threshold = kwargs.get("ligand_max_sim", 1)
            else:
                drug_distance = kwargs["ligand_dist"]
                drug_threshold = kwargs.get("ligand_max_dist", 1)
            drug_names = list(drugs.keys())
    else:
        drugs, drug_names, drug_weights, drug_similarity, drug_distance, drug_threshold = \
            None, None, None, None, None, 0

    return (
        (
            protein_names,
            proteins,
            protein_weights,
            protein_similarity,
            protein_distance,
            protein_threshold,
        ), (
            drug_names,
            drugs,
            drug_weights,
            drug_similarity,
            drug_distance,
            drug_threshold,
        ),
        inter,
    )


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


def read_pdb_folder(folder_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read in all PDB file from a folder and ignore non-PDB files.

    Args:
        folder_path: Path to the folder storing the PDB files

    Yields:
        Pairs of the PDB files name and the path to the file
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdb"):
            yield filename[:-4], os.path.join(folder_path, filename)


def count_inter(inter: List[Tuple[str, str]], mode: str) -> Generator[Tuple[str, int], None, None]:
    """
    Count interactions per protein or drug in a set of interactions.

    Args:
        inter: List of pairwise interactions of proteins and drugs
        mode: mode to read data for, either >protein> or >drug<

    Yields:
        Pairs of protein or drug names and the number of interactions they participate in
    """
    tmp = list(zip(*inter))
    mode = 0 if mode == "drugs" else 1
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
