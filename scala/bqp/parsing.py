import os.path
from typing import Tuple, Generator, Dict, List, Optional, Union, Any

import numpy as np

from scala.utils.utils import parse_fasta

ParseInfo = Tuple[
    Optional[List[str]],
    Optional[Dict[str, str]],
    Optional[Dict[str, float]],
    Optional[Union[np.ndarray, str]],
    float
]


def read_data(**kwargs) -> Tuple[ParseInfo, ParseInfo, Optional[List[Tuple[str, str]]]]:
    # TODO: Semantic checks of arguments
    inter = list(tuple(x) for x in read_csv(kwargs["inter"], kwargs["header"], kwargs["sep"])) if kwargs["inter"] else None

    # parse the proteins
    if kwargs["input"] is not None:
        if kwargs["input"].endswith(".fasta") or kwargs["input"].endswith(".fa"):
            proteins = parse_fasta(kwargs["input"])
        else:
            proteins = dict(read_csv(kwargs["input"], kwargs["header"], kwargs["sep"]))

        # parse the protein weights
        if kwargs["weight_file"] is not None:
            protein_weights = dict(read_csv(kwargs["weight_file"], kwargs["header"], kwargs["sep"]))
        elif inter is not None:
            protein_weights = dict(count_inter(inter, "prots"))
        else:
            protein_weights = np.zeros(len(proteins))  # TODO: Check if it needs to be np.ones(...)

        # parse the protein similarity measure
        if kwargs["prot_sim"] is None:
            protein_similarity = np.ones((len(proteins), len(proteins)))
            protein_names = list(proteins.keys())
        elif os.path.isfile(kwargs["prot_sim"]):
            protein_names, protein_similarity = read_similarity_file(kwargs["prot_sim"])
        else:
            protein_similarity = kwargs["prot_sim"]
            protein_names = list(proteins.keys())
        protein_min_sim = kwargs.get("prot_min_sim", 0)
    else:
        proteins, protein_names, protein_weights, protein_similarity, protein_min_sim = None, None, None, None, 0

    # parse molecules
    if kwargs["drugs"] is not None:
        drugs = dict(read_csv(kwargs["drugs"], kwargs["header"], kwargs["sep"]))

        # parse molecular weights
        if kwargs["drug_weights"] is not None:
            drug_weights = dict((x, float(v)) for x, v in read_csv(kwargs["drug_weights"], kwargs["header"], kwargs["sep"]))
        elif inter is not None:
            drug_weights = dict(count_inter(inter, "drugs"))
        else:
            drug_weights = None

        # parse molecular similarity
        if kwargs["drug_sim"] is None:
            drug_similarity = np.ones((len(drugs), len(drugs)))
            drug_names = list(drugs.keys())
        elif os.path.isfile(kwargs["drug_sim"]):
            drug_names, drug_similarity = read_similarity_file(kwargs["drug_sim"])
        else:
            drug_similarity = kwargs["drug_sim"]
            drug_names = list(drugs.keys())
        drug_min_sim = kwargs.get("drug_min_sim", 0)
    else:
        drugs, drug_names, drug_weights, drug_similarity, drug_min_sim = None, None, None, None, 0

    return (
        (
            protein_names,
            proteins,
            protein_weights,
            protein_similarity,
            protein_min_sim,
        ), (
            drug_names,
            drugs,
            drug_weights,
            drug_similarity,
            drug_min_sim,
        ),
        inter,
    )


def read_csv(filepath: str, header: bool = False, sep: str = "\t") -> Generator[Tuple[str, str], None, None]:
    with open(filepath, "r") as inter:
        for line in inter.readlines()[(1 if header else 0):]:
            yield line.strip().split(sep)[:2]


def count_inter(inter, mode) -> Generator[Tuple[str, int], None, None]:
    tmp = list(zip(*inter))
    mode = 0 if mode == "drugs" else 1
    keys = set(tmp[mode])
    for key in keys:
        yield key, tmp[mode].count(key)


def read_similarity_file(filepath: str, sep: str = "\t") -> Tuple[List[str], np.ndarray]:
    names = []
    similarities = []
    with open(filepath, "r") as data:
        for line in data.readlines():
            parts = line.strip().split(sep)
            names.append(parts[0])
            similarities.append([float(x) for x in parts[1:]])
    return names, np.array(similarities)
