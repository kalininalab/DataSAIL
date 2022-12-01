from typing import Tuple, Generator, Dict, List, Union, Optional

from sortedcontainers import SortedList


def read_data(**kwargs) -> Tuple[List[Tuple[str, str]], Optional[Dict], Dict, Optional[Dict], Dict]:
    if kwargs["inter"]:
        inter = list(read_csv(kwargs["inter"], kwargs["header"], kwargs["sep"]))
    else:
        raise ValueError()

    drugs = dict(read_csv(kwargs["drugs"], kwargs["header"], kwargs["sep"])) if kwargs["drugs"] else None
    drug_weights = dict(read_csv(kwargs["drug_weights"], kwargs["header"], kwargs["sep"])
                        if kwargs["drug_weights"] else count_inter(inter, "drugs"))

    proteins = dict(read_csv(kwargs["input"], kwargs["header"], kwargs["sep"])) if kwargs["input"] else None
    protein_weights = dict(read_csv(kwargs["weight_file"], kwargs["header"], kwargs["sep"])
                           if kwargs["weight_file"] else count_inter(inter, "prots"))

    # TODO: validate arguments on a semantic level
    # inter_drugs = set(d for p, d in data["interactions"])
    # drugs = set(data["drugs"])
    # weight_drugs = set(data["drug_weights"])

    return inter, drugs, drug_weights, proteins, protein_weights


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


def compute_drug_similarity(drugs):
    pass


def compute_protein_similarity(proteins):
    pass
