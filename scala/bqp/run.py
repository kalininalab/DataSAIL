import logging
import os
from typing import Dict, List, Tuple, Set

import numpy as np

from scala.bqp.algos.cluster_cold_double import solve_ccd_bqp
from scala.bqp.algos.cluster_cold_single import solve_ccs_bqp
from scala.bqp.algos.id_cold_double import solve_icd_bqp
from scala.bqp.algos.id_cold_single import solve_ics_bqp
from scala.bqp.clustering import cluster, cluster_interactions
from scala.bqp.parsing import read_data


def bqp_main(**kwargs):
    logging.info("Starting BQP solving")
    logging.info("Read data")

    (protein_names, proteins, protein_weights, protein_similarity, protein_distance, prot_threshold), \
        (drug_names, drugs, drug_weights, drug_similarity, drug_distance, drug_threshold), inter = read_data(**kwargs)
    drug_cluster_names, drug_cluster_map, drug_cluster_similarity, drug_cluster_distance, drug_cluster_weights = \
        cluster(drug_similarity, drug_distance, drugs, drug_weights, **kwargs)
    prot_cluster_names, prot_cluster_map, prot_cluster_similarity, prot_cluster_distance, prot_cluster_weights = \
        cluster(protein_similarity, drug_distance, proteins, protein_weights, **kwargs)

    output_inter, output_drugs, output_proteins = None, None, None

    logging.info("Split data")

    if kwargs["technique"] == "R":
        output_inter = sample_categorical(
            data=inter,
            splits=kwargs["splits"],
            names=kwargs["names"],
        )
    elif kwargs["technique"] == "ICD":
        solution = solve_ics_bqp(
            molecules=drug_names,
            weights=[drug_weights[d] for d in drug_names],
            limit=kwargs["limit"],
            splits=kwargs["splits"],
            names=kwargs["names"],
            max_sec=kwargs["max_sec"],
            max_sol=kwargs["max_sol"],
        )
        if solution is not None:
            output_drugs = solution
    if kwargs["technique"] == "ICP":
        solution = solve_ics_bqp(
            molecules=protein_names,
            weights=[protein_weights[p] for p in protein_names],
            limit=kwargs["limit"],
            splits=kwargs["splits"],
            names=kwargs["names"],
            max_sec=kwargs["max_sec"],
            max_sol=kwargs["max_sol"],
        )
        if solution is not None:
            output_proteins = solution
    if kwargs["technique"] == "IC":
        solution = solve_icd_bqp(
            drugs=drug_names,
            proteins=protein_names,
            inter=set(inter),
            limit=kwargs["limit"],
            splits=kwargs["splits"],
            names=kwargs["names"],
            max_sec=kwargs["max_sec"],
            max_sol=kwargs["max_sol"],
        )
        if solution is not None:
            output_inter, output_drugs, output_proteins = solution
    if kwargs["technique"] == "CCD":
        cluster_split = solve_ccs_bqp(
            clusters=drug_cluster_names,
            weights=[drug_cluster_weights[dc] for dc in drug_cluster_names],
            similarities=drug_cluster_similarity,
            distances=drug_cluster_distance,
            threshold=0.75,
            limit=kwargs["limit"],
            splits=kwargs["splits"],
            names=kwargs["names"],
            max_sec=kwargs["max_sec"],
            max_sol=kwargs["max_sol"],
        )
        if cluster_split is not None:
            output_inter, output_drugs = infer_interactions(cluster_split, set(inter))
    if kwargs["technique"] == "CCP":
        cluster_split = solve_ccs_bqp(
            clusters=prot_cluster_names,
            weights=[prot_cluster_weights[pc] for pc in prot_cluster_names],
            similarities=prot_cluster_similarity,
            distances=prot_cluster_distance,
            threshold=0.75,
            limit=kwargs["limit"],
            splits=kwargs["splits"],
            names=kwargs["names"],
            max_sec=kwargs["max_sec"],
            max_sol=kwargs["max_sol"],
        )
        if cluster_split is not None:
            output_inter, output_proteins = infer_interactions(cluster_split, set(inter))
    if kwargs["technique"] == "CC":
        cluster_inter = cluster_interactions(
            inter,
            drug_cluster_map,
            drug_cluster_names,
            prot_cluster_map,
            prot_cluster_names,
        )
        cluster_split = solve_ccd_bqp(
            drug_clusters=drug_cluster_names,
            drug_similarities=drug_cluster_similarity,
            drug_distances=drug_cluster_distance,
            drug_threshold=drug_threshold,
            prot_clusters=prot_cluster_names,
            prot_similarities=prot_cluster_similarity,
            prot_distances=prot_cluster_distance,
            prot_threshold=prot_threshold,
            inter=cluster_inter,
            limit=kwargs["limit"],
            splits=kwargs["splits"],
            names=kwargs["names"],
            max_sec=kwargs["max_sec"],
            max_sol=kwargs["max_sol"],
        )

        if cluster_split is not None:
            output_inter, output_drugs, output_proteins = cluster_split

    logging.info("Store results")

    if output_inter is None and output_drugs is not None and output_proteins is None:
        output_inter = [(d, p, output_drugs[d]) for d, p in inter]
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
        with open(os.path.join(kwargs["output"], "drugs.tsv"), "w") as stream:
            for drug, split in output_drugs.items():
                print(drug, split, sep=kwargs["sep"], file=stream)
                split_stats[split] += 1
        print("Drug distribution over splits:")
        print(stats_string(len(drugs), split_stats))

    if output_proteins is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "proteins.tsv"), "w") as stream:
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
