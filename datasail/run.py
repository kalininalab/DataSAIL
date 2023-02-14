import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np

from datasail.cluster.clustering import cluster
from datasail.reader.read import read_data
from .solver.solve import run_solver


def bqp_main(**kwargs) -> None:
    """
    Main routine of DataSAIL. Here the parsed input is aggregated into structures and then split and saved.

    Args:
        **kwargs: Parsed commandline arguments to DataSAIL.
    """
    start = time.time()
    logging.info("Starting BQP solving")
    logging.info("Read data")

    # read e-entities and f-entities in
    e_dataset, f_dataset, inter = read_data(**kwargs)

    # if required, cluster the input otherwise define the cluster-maps to be None
    if "C" == kwargs["technique"][0]:
        e_dataset = cluster(e_dataset, **kwargs)
        f_dataset = cluster(f_dataset, **kwargs)

    logging.info("Split data")
    # split the data into dictionaries mapping interactions, e-entities, and f-entities into the splits
    output_inter, output_e_entities, output_f_entities = run_solver(
        technique=kwargs["technique"],
        vectorized=kwargs["vectorized"],
        e_dataset=e_dataset,
        f_dataset=f_dataset,
        inter=inter,
        limit=kwargs["limit"],
        splits=kwargs["splits"],
        names=kwargs["names"],
        max_sec=kwargs["max_sec"],
        max_sol=kwargs["max_sol"],
    )

    logging.info("Store results")

    # infer interaction assignment from entity assignment if necessary and possible
    if inter is not None:
        if output_inter is None and output_e_entities is not None and output_f_entities is None:
            output_inter = [(e, f, output_e_entities[e]) for e, f in inter]
        elif output_inter is None and output_e_entities is None and output_f_entities is not None:
            output_inter = [(e, f, output_f_entities[f]) for e, f in inter]
        elif output_inter is None and output_e_entities is not None and output_f_entities is not None:
            output_inter = [
                (e, f, output_e_entities[e]) for e, f in inter if output_e_entities[e] == output_f_entities[f]
            ]

    # create the output folder to store the results in
    if not os.path.exists(kwargs["output"]):
        os.makedirs(kwargs["output"], exist_ok=True)

    # store interactions into a TSV file
    if output_inter is not None:
        split_stats = dict((n, 0) for n in kwargs["names"] + ["not selected"])
        with open(os.path.join(kwargs["output"], "inter.tsv"), "w") as stream:
            for e, f, split in output_inter:
                print(e, f, split, sep="\t", file=stream)
                split_stats[split] += 1
        print("Interaction-split statistics:")
        print(stats_string(len(inter), split_stats))

    # store entities into a TSV file
    for i, (entities, dataset) in enumerate([
        (output_e_entities, e_dataset), (output_f_entities, f_dataset)
    ]):
        if entities is not None:
            name = char2name(dataset.type)
            split_stats = dict((n, 0) for n in kwargs["names"])
            with open(os.path.join(kwargs["output"], f"{name}_{i + 1}.tsv"), "w") as stream:
                for e, split in entities.items():
                    print(e, split, sep="\t", file=stream)
                    split_stats[split] += 1
            print(name + " distribution over splits:")
            print(stats_string(len(dataset.names), split_stats))

    logging.info("BQP splitting finished and results stored.")
    logging.info(f"Total runtime: {time.time() - start:.5f}s")


def whatever(
        names: List[str], clusters: Dict[str, str], distances: Optional[np.ndarray], similarities: Optional[np.ndarray]
) -> None:
    """
    Compute and print some statistics.

    Args:
        names: names of the clusters to investigate
        clusters: mapping from entity name to cluster name
        distances: distance matrix between entities
        similarities: similarity matrix between entities
    """
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
    logging.debug("Some cluster statistics:")
    logging.debug(f"\tMin {metric_name}: {np.min(metric):.5f}")
    logging.debug(f"\tMax {metric_name}: {np.max(metric):.5f}")
    logging.debug(f"\tAvg {metric_name}: {np.average(metric):.5f}")
    logging.debug(f"\tMean {metric_name[:-1]}: {np.mean(metric):.5f}")
    logging.debug(f"\tVar {metric_name}: {np.var(metric):.5f}")
    if distances is not None:
        logging.debug(f"\tMaximal distance in same split: {val:.5f}")
        logging.debug(f"\t{(metric > val).sum() / len(metric) * 100:.2}% of distances are larger")
        logging.debug(f"\tMinimal distance between two splits: {val:.5f}")
        logging.debug(f"\t{(metric < val2).sum() / len(metric) * 100:.2}% of distances are smaller")
    else:
        logging.debug(f"Minimal similarity in same split {val:.5f}")
        logging.debug(f"\t{(metric < val).sum() / len(metric) * 100:.2}% of similarities are smaller")
        logging.debug(f"Maximal similarity between two splits {val:.5f}")
        logging.debug(f"\t{(metric > val).sum() / len(metric) * 100:.2}% of similarities are larger")


def stats_string(count: int, split_stats: Dict[str, float]):
    """
    Compute and print some statistics about the final splits.

    Args:
        count: number of totally split entities
        split_stats: mapping from split names to the number of elements in the split
    """
    output = ""
    for k, v in split_stats.items():
        output += f"\t{k:13}: {v:6}"
        if count > 0:
            output += f" {100 * v / count:>6.2f}%"
        else:
            output += f" {0:>6.2f}%"
        if k != "not selected":
            if (count - split_stats.get('not selected', 0)) > 0:
                output += f" {100 * v / (count - split_stats.get('not selected', 0)):>6.2f}%"
            else:
                output += f" {0:>6.2f}%"
        output += "\n"
    return output[:-1]


def char2name(c: chr) -> str:
    """
    Mapping from characters to type name in terms of entity type.

    Args:
        c: Single character name of the data type

    Returns:
        String telling the full name of the data type
    """
    if c == "P":
        return "Protein"
    if c == "M":
        return "Molecule"
    if c == "G":
        return "Genome"
    return "Other"
