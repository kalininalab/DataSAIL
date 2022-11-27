import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from sortedcontainers import SortedList

# from scala.cluster.wl_kernels.protein import smiles_to_grakel
# from scala.cluster.wl_kernels.wlk import run_wl_kernel
from scala.qilp.id_cold_single import solve_icx_qip
from scala.sat_split.sat_solvers.cluster_cold_single import solve_mkp_ilp_ccx
from scala.sat_split.sat_solvers.id_cold_double import solve_ic_sat
from scala.sat_split.sat_solvers.id_cold_single import solve_icx_sat
from scala.sat_split.read_data import read_data


def ilp_main(args):
    logging.info("Starting ILP solving")
    logging.info("Read data")

    data = read_data(args)
    output = {"inter": None, "drugs": None, "proteins": None}

    logging.info("Split data")

    if args.technique == "R":
        output["inter"] = sample_categorical(
            list(data["interactions"]),
            args.splits,
            args.names
        )
    if args.technique == "ICD":
        drug = SortedList(data["drugs"].keys())
        solution = solve_icx_qip(
            drug,
            [data["drug_weights"][d] for d in drug],
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
        if solution is not None:
            output["drugs"] = solution
    if args.technique == "ICP":
        prot = SortedList(data["proteins"].keys())
        solution = solve_icx_sat(
            prot,
            [data["prot_weights"][p] for p in prot],
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
        if solution is not None:
            output["proteins"] = solution
    if args.technique == "IC":
        solution = solve_ic_sat(
            list(data["drugs"].keys()),
            list(data["proteins"].keys()),
            set(tuple(x) for x in data["interactions"]),
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
        if solution is not None:
            output["inter"], output["drugs"], output["proteins"] = solution
    if args.technique == "CCD":
        clusters, cluster_sim, cluster_map = cluster(data["drugs"], "WLK")
        cluster_weights = []
        cluster_split = solve_mkp_ilp_ccx(
            clusters,
            cluster_weights,
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
    if args.technique == "CCP":
        pass
    if args.technique == "CC":
        pass

    logging.info("Store results")

    if output["inter"] is None and output["drugs"] is not None and output["proteins"] is None:
        output["inter"] = [(d, p, output["drugs"][d]) for d, p in data["interactions"]]
    if output["inter"] is None and output["drugs"] is None and output["proteins"] is not None:
        output["inter"] = [(d, p, output["proteins"][p]) for d, p in data["interactions"]]

    if output["inter"] is not None:
        split_stats = dict((n, 0) for n in args.names + ["not selected"])
        with open(os.path.join(args.output, "inter.tsv"), "w") as stream:
            for drug, prot, split in output["inter"]:
                print(drug, prot, split, sep=args.sep, file=stream)
                split_stats[split] += 1
        print("Interaction-split statistics:")
        print(stats_string(len(data["interactions"]), split_stats))

    if output["drugs"] is not None:
        split_stats = dict((n, 0) for n in args.names + ["not selected"])
        with open(os.path.join(args.output, "drug.tsv"), "w") as stream:
            for drug, split in output["drugs"].items():
                print(drug, split, sep=args.sep, file=stream)
                split_stats[split] += 1
        print("Drug distribution over splits:")
        print(stats_string(len(data["drugs"]), split_stats))

    if output["proteins"] is not None:
        split_stats = dict((n, 0) for n in args.names + ["not selected"])
        with open(os.path.join(args.output, "prot.tsv"), "w") as stream:
            for protein, split in output["proteins"].items():
                print(protein, split, sep=args.sep, file=stream)
                split_stats[split] += 1
        print("Protein distribution over splits:")
        print(stats_string(len(data["proteins"]), split_stats))

    logging.info("ILP splitting finished and results stored.")


def stats_string(count, split_stats):
    output = ""
    for k, v in split_stats.items():
        output += f"\t{k:13}: {v:6}"
        if count > 0:
            output += f" {100 * v / count:>6.2f}%"
        else:
            output += f" {0:>6.2f}%"
        if k != "not selected":
            if (count - split_stats['not selected']) > 0:
                output += f" {100 * v / (count - split_stats['not selected']):>6.2f}%"
            else:
                output += f" {0:>6.2f}%"
        output += "\n"
    return output[:-1]


def sample_categorical(
        data: List[Tuple[str, str]],
        splits: List[float],
        names: List[str],
):
    np.random.shuffle(data)

    def gen():
        for index in range(len(splits) - 1):
            yield data[int(sum(splits[:index]) * len(data)):int(sum(splits[:(index + 1)]) * len(data))]
        yield data[int(sum(splits[:-1]) * len(data)):]

    output = []
    for i, split in enumerate(gen()):
        output += [(d, p, names[i]) for d, p in split]
    return output


def cluster(mols: Dict[str, str], method: str) -> Tuple[int, Dict[str, int], List[List[int]]]:
    if method == "WLK":
        ids = list(mols.keys())
        # graphs = [smiles_to_grakel(mols[idx[0]]) for idx in ids]
        # cluster_sim = run_wl_kernel(graphs)
        cluster_map = dict((idx[0], i) for i, idx in enumerate(ids))
    else:
        raise ValueError("Unknown clustering method.")

    # return len(cluster_sim), cluster_map, cluster_sim
    return 0, cluster_map, None
