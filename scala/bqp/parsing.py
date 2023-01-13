import os.path
from typing import Tuple, Generator, Dict, List, Optional

import numpy as np

from scala.utils.utils import parse_fasta


def read_data(**kwargs):# -> Tuple[List[Tuple[str, str]], Optional[Dict], Dict, Optional[Dict], Dict]:
    # TODO: Method signature
    # TODO: Semantic checks of arguments

    inter = list(read_csv(kwargs["inter"], kwargs["header"], kwargs["sep"])) if kwargs["inter"] else None

    # parse the proteins
    if not kwargs["input"]:
        proteins = None
    elif kwargs["input"].endswith(".fasta") or kwargs["input"].endswith(".fa"):
        proteins = parse_fasta(kwargs["input"])
    else:
        proteins = dict(read_csv(kwargs["input"], kwargs["header"], kwargs["sep"]))

    # parse the protein weights
    if kwargs["weight_file"]:
        protein_weights = dict(read_csv(kwargs["weight_file"], kwargs["header"], kwargs["sep"]))
    elif inter:
        protein_weights = dict(count_inter(inter, "prots"))
    else:
        protein_weights = np.zeros(len(proteins))  # TODO: Check if it needs to be np.ones(...)

    # parse the protein similarity measure
    if not kwargs["prot_sim"]:
        protein_similarity = np.ones((len(proteins), len(proteins)))
        protein_names = list(proteins.keys())
    elif os.path.isfile(kwargs["prot_sim"]):
        protein_names, protein_similarity = read_similarity_file(kwargs["prot_sim"])
    else:
        protein_similarity = kwargs["prot_sim"]
        protein_names = list(proteins.keys())

    # parse molecules
    if kwargs["drugs"]:
        drugs = dict(read_csv(kwargs["drugs"], kwargs["header"], kwargs["sep"]))
    else:
        drugs = None

    # parse molecular weights
    if drugs and kwargs["drug_weights"]:
        drug_weights = dict(read_csv(kwargs["drug_weights"], kwargs["header"], kwargs["sep"]))
    elif drugs and inter:
        drug_weights = dict(count_inter(inter, "drugs"))
    else:
        drug_weights = None

    # parse molecular similarity
    if drugs and not kwargs["drug_sim"]:
        drug_similarity = np.ones((len(drugs), len(drugs)))
        drug_names = list(drugs.keys())
    elif drugs and os.path.isfile(kwargs["prot_sim"]):
        drug_names, drug_similarity = read_similarity_file(kwargs["drug_sim"])
    else:
        drug_similarity = kwargs["drug_sim"]
        drug_names = list(drugs.keys())

    return (
        (
            protein_names,
            proteins,
            protein_weights,
            protein_similarity,
            kwargs.get("prot_min_sim", 0),
        ), (
            drug_names,
            drugs,
            drug_weights,
            drug_similarity,
            kwargs.get("drug_min_sim", 0),
        ),
        inter
    )


def read_csv(filepath: str, header: bool = False, sep: str = "\t") -> Generator[Tuple[str, str], None, None]:
    with open(filepath, "r") as inter:
        for line in inter.readlines()[(1 if header else 0):]:
            yield line.strip().split(sep)[:2]


def count_inter(inter, mode):
    tmp = list(zip(*inter))
    mode = 0 if mode == "drugs" else 1
    keys = set(tmp[mode])
    for key in keys:
        yield key, tmp[mode].count(key)


def read_similarity_file(filepath: str, sep: str = "\t") -> object:
    names = []
    similarities = []
    with open(filepath, "r") as data:
        for line in data.readlines():
            parts = line.strip().split(sep)
            names.append(parts[0])
            similarities.append([float(x) for x in parts[1:]])
    return names, similarities
