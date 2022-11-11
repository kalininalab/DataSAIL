from typing import Tuple, Generator, Dict, List, Union

from sortedcontainers import SortedList


def read_data(args) -> Dict[str, Union[List, Dict]]:
    data = {}

    if args.inter is not None:
        data["interactions"] = SortedList(read_csv(args.inter, args.header, args.sep))

    if args.drugs is not None:
        data["drugs"] = dict(read_csv(args.drugs, args.header, args.sep))
    data["drug_weights"] = dict(read_csv(args.drug_weights, args.header, args.sep) if args.drug_weights is not None
                                else count_inter(data["interactions"], "drugs"))
    if args.technique in {}:
        data["drug_sim"] = compute_drug_similarity(data["drugs"])

    if args.input is not None:
        data["proteins"] = dict(read_csv(args.input, args.header, args.sep))
    data["prot_weights"] = dict(read_csv(args.weight_file, args.header, args.sep) if args.weight_file is not None
                                else count_inter(data["interactions"], "prots"))
    if args.technique in {}:
        data["prot_sim"] = compute_protein_similarity(data["proteins"])

    # validate arguments on a semantic level
    inter_drugs = set(d for p, d in data["interactions"])
    drugs = set(data["drugs"])
    weight_drugs = set(data["drug_weights"])

    return data


def read_csv(filepath: str, header: bool = False, sep: str = "\t") -> Generator[Tuple[int, int], None, None]:
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
