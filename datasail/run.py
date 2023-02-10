import logging
import os
import time
from typing import Dict, List, Tuple, Set

import numpy as np

from .clustering import cluster
from datasail.reader.read import read_data
from .solver.solve import run_solver


def bqp_main(**kwargs) -> None:
    start = time.time()
    logging.info("Starting BQP solving")
    logging.info("Read data")

    (e_type, (e_names, e_data, e_weights, e_similarity, e_distance, e_threshold)), \
        (f_type, (f_names, f_data, f_weights, f_similarity, f_distance, f_threshold)), inter = read_data(**kwargs)
    if "C" == kwargs["technique"][0]:
        e_names, e_cluster_map, e_similarity, e_distance, e_weights = \
            cluster(e_similarity, e_distance, e_data, e_weights, **kwargs)
        f_names, f_cluster_map, f_similarity, f_distance, f_weights = \
            cluster(f_similarity, f_distance, f_data, f_weights, **kwargs)
    else:
        e_cluster_map, f_cluster_map = None, None

    logging.info("Split data")
    output_inter, output_e_entities, output_f_entities = run_solver(
        technique=kwargs["technique"],
        vectorized=kwargs["vectorized"],
        e_names=e_names,
        e_cluster_map=e_cluster_map,
        e_weights=e_weights,
        e_similarities=e_similarity,
        e_distances=e_distance,
        e_threshold=e_threshold,
        f_names=f_names,
        f_cluster_map=f_cluster_map,
        f_weights=f_weights,
        f_similarities=f_similarity,
        f_distances=f_distance,
        f_threshold=f_threshold,
        inter=inter,
        limit=kwargs["limit"],
        splits=kwargs["splits"],
        names=kwargs["names"],
        max_sec=kwargs["max_sec"],
        max_sol=kwargs["max_sol"],
    )

    logging.info("Store results")

    if inter is not None:
        if output_inter is None and output_e_entities is not None and output_f_entities is None:
            output_inter = [(e, f, output_e_entities[e]) for e, f in inter]
        elif output_inter is None and output_e_entities is None and output_f_entities is not None:
            output_inter = [(e, f, output_f_entities[f]) for e, f in inter]
        elif output_inter is None and output_e_entities is not None and output_f_entities is not None:
            output_inter = [(e, f, output_e_entities[e]) for e, f in inter if output_e_entities[e] == output_f_entities[f]]

    if not os.path.exists(kwargs["output"]):
        os.makedirs(kwargs["output"], exist_ok=True)

    if output_inter is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "inter.tsv"), "w") as stream:
            for e, f, split in output_inter:
                print(e, f, split, sep="\t", file=stream)
                split_stats[split] += 1
        print("Interaction-split statistics:")
        print(stats_string(len(inter), split_stats))

    if output_e_entities is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "drugs.tsv"), "w") as stream:
            for e, split in output_e_entities.items():
                print(e, split, sep="\t", file=stream)
                split_stats[split] += 1
        print("Drug distribution over splits:")
        print(stats_string(len(e_names), split_stats))

    if output_f_entities is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "proteins.tsv"), "w") as stream:
            for f, split in output_f_entities.items():
                print(f, split, sep="\t", file=stream)
                split_stats[split] += 1
        print("Protein distribution over splits:")
        print(stats_string(len(f_names), split_stats))

    logging.info("BQP splitting finished and results stored.")
    logging.info(f"Total runtime: {time.time() - start:.5f}s")


def whatever(names: List[str], clusters: Dict[str, str], distances: np.ndarray, similarities: np.ndarray):
    # TODO: optimize this for runtime
    if distances is not None:
        val = float("-inf")
        val2 = float("inf")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if clusters[names[i]] == clusters[names[j]]:
                    val = max(val, distances[i, j])
                else:
                    val2 = min(val2, distances[i, j])
    else:
        val = float("inf")
        val2 = float("-inf")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if clusters[names[i]] == clusters[names[j]]:
                    val = min(val, similarities[i, j])
                else:
                    val2 = max(val, similarities[i, j])

    metric_name = "distance   " if distances is not None else "similarity "
    metric = distances.flatten() if distances is not None else similarities.flatten()
    logging.info("Some cluster statistics:")
    logging.info(f"\tMin {metric_name}: {np.min(metric):.5f}")
    logging.info(f"\tMax {metric_name}: {np.max(metric):.5f}")
    logging.info(f"\tAvg {metric_name}: {np.average(metric):.5f}")
    logging.info(f"\tMean {metric_name[:-1]}: {np.mean(metric):.5f}")
    logging.info(f"\tVar {metric_name}: {np.var(metric):.5f}")
    if distances is not None:
        logging.info(f"\tMaximal distance in same split: {val:.5f}")
        logging.info(f"\t{(metric > val).sum() / len(metric) * 100:.2}% of distances are larger")
        logging.info(f"\tMinimal distance between two splits: {val:.5f}")
        logging.info(f"\t{(metric < val2).sum() / len(metric) * 100:.2}% of distances are smaller")
    else:
        logging.info(f"Minimal similarity in same split {val:.5f}")
        logging.info(f"\t{(metric < val).sum() / len(metric) * 100:.2}% of similarities are smaller")
        logging.info(f"Maximal similarity between two splits {val:.5f}")
        logging.info(f"\t{(metric > val).sum() / len(metric) * 100:.2}% of similarities are larger")


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
        first: bool = False,
) -> List[Tuple[str, str, str]]:
    def get_key(e, f):
        return e if first else f

    return [(e, f, molecule_split[get_key(e, f)]) for e, f in inter]
