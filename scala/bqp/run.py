import logging
import os
from typing import Dict, List, Tuple, Set

import numpy as np

from scala.bqp.algos.cluster_cold_double import solve_cc_iqp
from scala.bqp.algos.cluster_cold_single import solve_ccx_iqp
from scala.bqp.algos.id_cold_double import solve_ic_iqp
from scala.bqp.algos.id_cold_single import solve_icx_iqp
from scala.cluster.wl_kernels.protein import smiles_to_grakel
from scala.cluster.wl_kernels.wlk import run_wl_kernel
from scala.bqp.read_data import read_data


def bqp_main(**kwargs):
    logging.info("Starting ILP solving")
    logging.info("Read data")

    inter, drugs, drug_weights, proteins, protein_weights = read_data(**kwargs)
    output_inter, output_drugs, output_proteins = None, None, None

    logging.info("Split data")

    if kwargs["technique"] == "R":
        output_inter = sample_categorical(
            inter,
            kwargs["splits"],
            kwargs["names"],
        )
    elif kwargs["technique"] == "ICD":
        drug = list(drugs.keys())
        solution = solve_icx_iqp(
            drug,
            [drug_weights[d] for d in drug],
            kwargs["limit"],
            kwargs["splits"],
            kwargs["names"],
            kwargs["max_sec"],
            kwargs["max_sol"],
        )
        if solution:
            output_drugs = solution
    if kwargs["technique"] == "ICP":
        prot = list(proteins.keys())
        solution = solve_icx_iqp(
            prot,
            [protein_weights[p] for p in prot],
            kwargs["limit"],
            kwargs["splits"],
            kwargs["names"],
            kwargs["max_sec"],
            kwargs["max_sol"],
        )
        if solution:
            output_proteins = solution
    if kwargs["technique"] == "IC":
        solution = solve_ic_iqp(
            list(drugs.keys()),
            list(proteins.keys()),
            set(inter),
            kwargs["limit"],
            kwargs["splits"],
            kwargs["names"],
            kwargs["max_sec"],
            kwargs["max_sol"],
        )
        if solution is not None:
            output_inter, output_drugs, output_proteins = solution
    if kwargs["technique"] == "CCD":
        clusters, cluster_map, cluster_sim = cluster(drugs, "WLK")
        cluster_weights = []
        cluster_split = solve_ccx_iqp(
            list(range(clusters)),
            cluster_weights,
            cluster_sim,
            0.75,
            kwargs["limit"],
            kwargs["splits"],
            kwargs["names"],
            kwargs["max_sec"],
            kwargs["max_sol"],
        )
        if cluster_split:
            output_inter, output_drugs = infer_interactions(cluster_split, set(inter))
    if kwargs["technique"] == "CCP":
        clusters, cluster_map, cluster_sim = cluster(proteins, "WLK")
        cluster_weights = []
        cluster_split = solve_ccx_iqp(
            list(range(clusters)),
            cluster_weights,
            cluster_sim,
            0.75,
            kwargs["limit"],
            kwargs["splits"],
            kwargs["names"],
            kwargs["max_sec"],
            kwargs["max_sol"],
        )
        if cluster_split:
            output_inter, output_proteins = infer_interactions(cluster_split, set(inter))
    if kwargs["technique"] == "CC":
        drug_clusters, drug_cluster_map, drug_cluster_sim = cluster(drugs, "WLK")
        prot_clusters, prot_cluster_map, prot_cluster_sim = cluster(proteins, "WLK")
        cluster_inter = cluster_interactions(inter, drug_clusters, drug_cluster_sim, prot_clusters, prot_cluster_sim)
        cluster_split = solve_cc_iqp(
            list(range(drug_clusters)),
            [],
            drug_cluster_sim,
            0.75,
            list(range(prot_clusters)),
            [],
            prot_cluster_sim,
            0.75,
            cluster_inter,
            kwargs["limit"],
            kwargs["splits"],
            kwargs["names"],
            kwargs["max_sec"],
            kwargs["max_sol"],
        )
        if cluster_split:
            output_inter, output_drugs, output_proteins = cluster_split

    logging.info("Store results")

    if output_inter is None and output_inter is not None and output_proteins is None:
        output_inter = [(d, p, output_inter[d]) for d, p in inter]
    elif output_inter is None and output_drugs is None and output_proteins is not None:
        output_inter = [(d, p, output_proteins[p]) for d, p in inter]

    if not os.path.exists(kwargs["output"]):
        os.makedirs(kwargs["output"], exist_ok=True)

    if output_inter is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "inter.tsv"), "w") as stream:
            for drug, prot, split in output_inter:
                print(drug, prot, split, sep=kwargs["sep"], file=stream)
                split_stats[split] += 1
        print("Interaction-split statistics:")
        print(stats_string(len(inter), split_stats))

    if output_drugs is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "drug.tsv"), "w") as stream:
            for drug, split in output_drugs.items():
                print(drug, split, sep=kwargs["sep"], file=stream)
                split_stats[split] += 1
        print("Drug distribution over splits:")
        print(stats_string(len(drugs), split_stats))

    if output_proteins is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "prot.tsv"), "w") as stream:
            for protein, split in output_proteins.items():
                print(protein, split, sep=kwargs["sep"], file=stream)
                split_stats[split] += 1
        print("Protein distribution over splits:")
        print(stats_string(len(proteins), split_stats))

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


def cluster_interactions(
        inter,
        num_drug_clusters,
        drug_cluster_map,
        num_prot_clusters,
        prot_cluster_map
) -> List[List[int]]:
    output = [[0 for _ in range(num_prot_clusters)] for _ in range(num_drug_clusters)]

    for drug, protein in inter:
        output[drug_cluster_map[drug]][prot_cluster_map[protein]] += 1

    return output


def infer_interactions(
        molecule_split: Dict[str, str],
        inter: Set[Tuple[str, str]],
        protein: bool = False,
) -> List[Tuple[str, str, str]]:
    def get_key(d, p):
        return p if protein else d

    return [(drug, protein, molecule_split[get_key(drug, protein)]) for drug, protein in inter]


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
        graphs = [smiles_to_grakel(mols[idx[0]]) for idx in ids]
        cluster_sim = run_wl_kernel(graphs)
        cluster_map = dict((idx[0], i) for i, idx in enumerate(ids))
    else:
        raise ValueError("Unknown clustering method.")

    return len(cluster_sim), cluster_map, cluster_sim
