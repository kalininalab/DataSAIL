import math
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER


def report(
        techniques: Set[str],
        e_dataset: DataSet,
        f_dataset: DataSet,
        e_name_split_map: Dict[str, Dict[str, str]],
        f_name_split_map: Dict[str, Dict[str, str]],
        e_cluster_split_map: Dict[str, Dict[str, str]],
        f_cluster_split_map: Dict[str, Dict[str, str]],
        inter_split_map: Dict[str, List[Tuple[str, str, str]]],
        output_dir: str,
        split_names: List[str],
) -> None:
    # create the output folder to store the results in
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for t in techniques:
        technique, mode = t[:3], t[-1]
        if mode.isupper():
            mode = None

        save_dir = os.path.join(output_dir, t)
        os.makedirs(save_dir, exist_ok=True)

        if t in inter_split_map:
            save_inter_assignment(save_dir, inter_split_map[t])

        if e_dataset.type is not None \
                and ((mode is not None and mode != "f") or technique[-1] == "D") \
                and technique in e_name_split_map:
            individual_report(save_dir, e_dataset, e_name_split_map, e_cluster_split_map, technique, split_names)

        if f_dataset.type is not None \
                and ((mode is not None and mode != "e") or technique[-1] == "D") \
                and technique in f_name_split_map:
            individual_report(save_dir, f_dataset, f_name_split_map, f_cluster_split_map, technique, split_names)


def individual_report(
        save_dir: str,
        dataset: DataSet,
        name_split_map: Dict[str, Dict[str, str]],
        cluster_split_map: Dict[str, Dict[str, str]],
        technique: str,
        split_names: List[str],
):
    """
    Create all the report files for one dataset and one technique.

    Args:
        save_dir: Directory to store the files in
        dataset: Dataset to store the results from
        name_split_map: Mapping of sample ids to splits
        cluster_split_map: Mapping from cluster names to splits
        technique: Technique to treat here
        split_names: Names of the splits
    """
    save_assignment(save_dir, dataset, name_split_map.get(technique, None))
    if technique[0] == "C":
        save_clusters(save_dir, dataset)
        save_t_sne(save_dir, dataset, name_split_map.get(technique, None), cluster_split_map.get(technique, None),
                   split_names)
        save_cluster_hist(save_dir, dataset)
    split_counts = dict((n, 0) for n in split_names)
    for name in dataset.names:
        split_counts[name_split_map[technique][name]] += dataset.weights.get(name, 0)
    print(stats_string(sum(dataset.weights.values()), split_counts))


def save_inter_assignment(save_dir: str, inter_split_map: Optional[List[Tuple[str, str, str]]]):
    """
    Save the assignment of interactions to splits in a TSV file.

    Args:
        save_dir: Directory to store the file in.
        inter_split_map: Mapping from interactions to the splits
    """
    if inter_split_map is None:
        return
    with open(os.path.join(save_dir, "inter.tsv"), "w") as output:
        for e, f, s in inter_split_map:
            print(e, f, s, sep="\t", file=output)


def save_assignment(save_dir: str, dataset: DataSet, name_split_map: Optional[Dict[str, str]]):
    """
    Save an assignment from data points to splits.

    Args:
        save_dir: Directory to store the file in
        dataset: Dataset to store the sample assignment from
        name_split_map: Mapping from sample ids to their assigned splits.
    """
    if name_split_map is None:
        return
    with open(os.path.join(
            save_dir, f"{char2name(dataset.type)}_{dataset.location.split('/')[-1].split('.')[0]}_splits.tsv"
    ), "w") as output:
        for e, s in name_split_map.items():
            print(e, s, sep="\t", file=output)


def save_clusters(save_dir: str, dataset: DataSet):
    """
    Save a clustering to a TSV file. The clustering is the mapping from data points to cluster representatives or names.

    Args:
        save_dir: Directory to store the file in
        dataset: Dataset to store the cluster assignment from
    """
    if dataset.cluster_map is None:
        return
    with open(os.path.join(
        save_dir, f"{char2name(dataset.type)}_{dataset.location.split('/')[-1].split('.')[0]}_clusters.tsv"
    ), "w") as output:
        for e, c in dataset.cluster_map.items():
            print(e, c, sep="\t", file=output)


def save_t_sne(
        save_dir: str,
        dataset: DataSet,
        name_split_map: Dict[str, str],
        cluster_split_map: Dict[str, str],
        split_names: List[str]
):
    """
    Compute and save the tSNE-plots for the splits visualizing the cluster assignments in 2D space.

    Args:
        save_dir: Directory to store the plot in
        dataset: Dataset to visualize
        name_split_map: Mapping from entity names to their splits
        cluster_split_map: Mapping from cluster representatives to their splits
        split_names: Names of the splits
    """
    if any(x is not None for x in [
        dataset.similarity, dataset.distance, dataset.cluster_similarity, dataset.cluster_distance
    ]):
        if (isinstance(dataset.similarity, np.ndarray) or isinstance(dataset.distance, np.ndarray)) \
                and name_split_map is not None:
            distance = dataset.distance if dataset.distance is not None else 1 - dataset.similarity
            t_sne(dataset.names, distance, name_split_map, split_names, os.path.join(
                save_dir, f"{char2name(dataset.type)}_{dataset.location.split('/')[-1].split('.')[0]}_splits.png"
            ))

        if (isinstance(dataset.cluster_similarity, np.ndarray) or isinstance(dataset.cluster_distance, np.ndarray)) \
                and cluster_split_map is not None:
            distance = dataset.cluster_distance if dataset.cluster_distance is not None \
                else 1 - dataset.cluster_similarity
            t_sne(dataset.cluster_names, distance, cluster_split_map, split_names, os.path.join(
                save_dir, f"{char2name(dataset.type)}_{dataset.location.split('/')[-1].split('.')[0]}_clusters.png"
            ))


def t_sne(names, distances, name_split_map: Dict[str, str], split_names: List[str], output_file_name: str):
    """
    Plot a tSNE embedding of the clusters and how they are assigned to clusters.

    Args:
        names: List of names in the dataset
        distances: a distance matrix to use for tSNE
        name_split_map: Mapping from names to splits
        split_names: names of the splits
        output_file_name: filepath to store the tSNE plot at
    """
    # compute t-SNE embeddings
    embeds = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        perplexity=max(min(math.sqrt(len(distances)), 50), 5),
        random_state=42,
    ).fit_transform(distances)

    # plot everything
    split_masks = np.zeros((len(split_names), len(names)))
    for i, name in enumerate(names):
        split_masks[split_names.index(name_split_map[name]), i] = 1
    for i, n in enumerate(split_names):
        plt.scatter(embeds[split_masks[i, :] == 1, 0], embeds[split_masks[i, :] == 1, 1], s=10, label=n)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(output_file_name)
    plt.clf()


def save_cluster_hist(save_dir, dataset):
    clusters = set(dataset.cluster_map.values())
    clusters = dict((c, i) for i, c in enumerate(clusters))
    counts = [0] * len(clusters)
    for n, c in dataset.cluster_map.items():
        counts[clusters[c]] += 1
    sizes = [0] * (max(counts) + 1)
    for c in counts:
        sizes[c] += 1
    plt.hist(counts)
    plt.savefig(os.path.join(save_dir, f"{char2name(dataset.type)}_{dataset.location.split('/')[-1].split('.')[0]}_cluster_hist.png"))
    plt.clf()


def whatever(
        names: List[str], clusters: Dict[str, str], distances: Optional[np.ndarray], similarities: Optional[np.ndarray]
) -> None:
    """
    Compute and print some statistics.

    Args:
        names: Names of the clusters to investigate
        clusters: Mapping from entity name to cluster name
        distances: Distance matrix between entities
        similarities: Similarity matrix between entities
    """
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
    LOGGER.info("Some cluster statistics:")
    LOGGER.info(f"\tMin {metric_name}: {np.min(metric):.5f}")
    LOGGER.info(f"\tMax {metric_name}: {np.max(metric):.5f}")
    LOGGER.info(f"\tAvg {metric_name}: {np.average(metric):.5f}")
    LOGGER.info(f"\tMean {metric_name[:-1]}: {np.mean(metric):.5f}")
    LOGGER.info(f"\tVar {metric_name}: {np.var(metric):.5f}")
    if distances is not None:
        LOGGER.info(f"\tMaximal distance in same split: {val:.5f}")
        LOGGER.info(f"\t{(metric > val).sum() / len(metric) * 100:.2}% of distances are larger")
        LOGGER.info(f"\tMinimal distance between two splits: {val:.5f}")
        LOGGER.info(f"\t{(metric < val2).sum() / len(metric) * 100:.2}% of distances are smaller")
    else:
        LOGGER.info(f"Minimal similarity in same split {val:.5f}")
        LOGGER.info(f"\t{(metric < val).sum() / len(metric) * 100:.2}% of similarities are smaller")
        LOGGER.info(f"Maximal similarity between two splits {val:.5f}")
        LOGGER.info(f"\t{(metric > val).sum() / len(metric) * 100:.2}% of similarities are larger")


def stats_string(count: int, split_stats: Dict[str, float]):
    """
    Compute and print some statistics about the final splits.

    Args:
        count: Number of totally split entities
        split_stats: Mapping from split names to the number of elements in the split
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
