import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from datasail.reader.utils import DataSet, DictMap
from datasail.settings import LOGGER, NOT_ASSIGNED, DIM_2, MODE_F, MODE_E, SRC_CL


def report(
        techniques: Set[str],
        e_dataset: DataSet,
        f_dataset: DataSet,
        e_name_split_map: DictMap,
        f_name_split_map: DictMap,
        e_cluster_split_map: DictMap,
        f_cluster_split_map: DictMap,
        inter_split_map: Dict[str, List[Dict[Tuple[str, str], str]]],
        runs: int,
        output_dir: Path,
        split_names: List[str],
) -> None:
    """
    Central entrypoint to create reports on the computed splits. This stores t-SNE plots, histograms, cluster- and
    split assignments for every split where the according reporting applies.

    Args:
        techniques: Set of techniques used to split the data
        e_dataset: First dataset
        f_dataset: Second dataset
        e_name_split_map: Mapping of splits to a mapping of names to splits for first dataset
        f_name_split_map: Mapping of splits to a mapping of names to splits for second dataset
        e_cluster_split_map: Mapping of splits to a mapping of names to cluster names for first dataset
        f_cluster_split_map: Mapping of splits to a mapping of names to cluster names for second dataset
        inter_split_map: Mapping of splits to a mapping of interactions to splits
        runs:
        output_dir: Output directory where to store the results
        split_names: Names of the splits
    """
    # create the output folder to store the results in
    output_dir.mkdir(parents=True, exist_ok=True)

    for t in techniques:
        for run in range(runs):
            mode = t[-1]
            if mode.isupper():
                mode = None

            # create output directory for reports of this split
            folder_name = t
            if runs > 1:
                folder_name += f"_{run + 1}"
            save_dir = output_dir / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # save mapping of interactions for this split if applicable
            if t in inter_split_map:
                save_inter_assignment(save_dir, inter_split_map[t][run])

            # Compile report for first dataset if applies for this split
            if e_dataset.type is not None and ((mode is not None and mode == MODE_E) or t[-1] == DIM_2) and \
                    t in e_name_split_map:
                individual_report(
                    save_dir,
                    e_dataset,
                    dict((t, e_name_split_map[t][run]) for t in e_name_split_map),
                    dict((t, e_cluster_split_map[t][run]) for t in e_cluster_split_map),
                    t,
                    split_names
                )

            # Compile report for second dataset if applies for this split
            if f_dataset.type is not None and ((mode is not None and mode == MODE_F) or t[-1] == DIM_2) \
                    and t in f_name_split_map:
                individual_report(
                    save_dir,
                    f_dataset,
                    dict((t, f_name_split_map[t][run]) for t in f_name_split_map),
                    dict((t, f_cluster_split_map[t][run]) for t in f_cluster_split_map),
                    t,
                    split_names
                )


def individual_report(
        save_dir: Path,
        dataset: DataSet,
        name_split_map: Dict[str, Dict[str, str]],
        cluster_split_map: Dict[str, Dict[str, str]],
        technique: str,
        split_names: List[str],
) -> None:
    """
    Create all the report files for one dataset and one technique.

    Args:
        save_dir: Directory to store the files in.
        dataset: Dataset to store the results from.
        name_split_map: Mapping of sample ids to splits.
        cluster_split_map: Mapping from cluster names to splits.
        technique: Technique to treat here.
        split_names: Names of the splits.
    """
    # Save assignment of names to splits
    save_assignment(save_dir, dataset, name_split_map.get(technique, None))

    # Save clustering-related reports
    if technique[0] == SRC_CL:
        save_clusters(save_dir, dataset)
        save_t_sne(save_dir, dataset, name_split_map.get(technique, None), cluster_split_map.get(technique, None),
                   split_names)
        save_cluster_hist(save_dir, dataset)

    # print statistics on how the sizes of the splits are distributed
    # split_counts = dict((n, 0) for n in split_names)
    # print(name_split_map[technique])
    # for name in dataset.names:
    #     split_counts[name_split_map[technique][name]] += dataset.weights.get(name, 0)
    # print(stats_string(sum(dataset.weights.values()), split_counts))


def save_inter_assignment(save_dir: Path, inter_split_map: Optional[Dict[Tuple[str, str], str]]) -> None:
    """
    Save the assignment of interactions to splits in a TSV file.

    Args:
        save_dir: Directory to store the file in.
        inter_split_map: Mapping from interactions to the splits
    """
    if inter_split_map is None:
        return

    pd.DataFrame(
        [(x1, x2, x3) for (x1, x2), x3 in inter_split_map.items()],
        columns=["E_ID", "F_ID", "Split"],
    ).to_csv(save_dir / "inter.tsv", sep="\t", columns=["E_ID", "F_ID", "Split"], index=False)


def save_assignment(save_dir: Path, dataset: DataSet, name_split_map: Optional[Dict[str, str]]) -> None:
    """
    Save an assignment from data points to splits.

    Args:
        save_dir: Directory to store the file in
        dataset: Dataset to store the sample assignment from
        name_split_map: Mapping from sample ids to their assigned splits.
    """
    if name_split_map is None:
        return

    filepath = save_dir / f"{char2name(dataset.type)}_{dataset.get_name()}_splits.tsv"

    pd.DataFrame(
        [(x1, name_split_map.get(x2, "")) for x1, x2 in dataset.id_map.items()],
        columns=["ID", "Split"]
    ).to_csv(filepath, sep="\t", columns=["ID", "Split"], index=False)


def save_clusters(save_dir: Path, dataset: DataSet) -> None:
    """
    Save a clustering to a TSV file. The clustering is the mapping from data points to cluster representatives or names.

    Args:
        save_dir: Directory to store the file in
        dataset: Dataset to store the cluster assignment from
    """
    if dataset.cluster_map is None:
        return
    filepath = save_dir / f"{char2name(dataset.type)}_{dataset.get_name()}_clusters.tsv"

    pd.DataFrame(
        [(x1, dataset.cluster_map.get(x2, "")) for x1, x2 in dataset.id_map.items()],
        columns=["ID", "Cluster_ID"],
    ).to_csv(filepath, sep="\t", columns=["ID", "Cluster_ID"], index=False)


def save_t_sne(
        save_dir: Path,
        dataset: DataSet,
        name_split_map: Dict[str, str],
        cluster_split_map: Dict[str, str],
        split_names: List[str]
) -> None:
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
            save_matrix_tsne(dataset.similarity, dataset.distance, dataset.names, dataset, name_split_map, split_names,
                             save_dir, "splits")

        if (isinstance(dataset.cluster_similarity, np.ndarray) or isinstance(dataset.cluster_distance, np.ndarray)) \
                and cluster_split_map is not None:
            save_matrix_tsne(dataset.cluster_similarity, dataset.cluster_distance, dataset.cluster_names, dataset,
                             cluster_split_map, split_names, save_dir, "clusters")


def save_matrix_tsne(
        similarities: Optional[np.ndarray],
        distances: Optional[np.ndarray],
        names: List[str],
        dataset: DataSet,
        entity_split_map: Dict[str, str],
        split_names: List[str],
        save_dir: Path,
        postfix: str
) -> None:
    """
    Plot a tSNE embedding of the clusters and how they are assigned to clusters.

    Args:
        similarities: a similarity matrix to convert into distance before running tSNE
        distances: a distance matrix to use for tSNE
        names: List of names in the dataset
        dataset: The dataset to run tSNE for. This is used to get additional information on the dataset
        entity_split_map: Mapping from names to splits
        split_names: names of the splits
        save_dir: Directory where to save the computed tSNE plot
        postfix: Postfix for the filename
    """
    distances = distances if distances is not None else 1 - similarities
    output_file_name = save_dir / f"{char2name(dataset.type)}_{dataset.get_name()}_{postfix}.png"
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
        split_masks[split_names.index(entity_split_map[name]), i] = 1
    for i, n in enumerate(split_names):
        plt.scatter(embeds[split_masks[i, :] == 1, 0], embeds[split_masks[i, :] == 1, 1], s=10, label=n)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(output_file_name)
    plt.clf()


def save_cluster_hist(save_dir: Path, dataset: DataSet) -> None:
    """
    Store visualization of cluster sizes.

    Args:
        save_dir: Directory to store the image in
        dataset: Dataset to compute the visualization for
    """
    clusters = set(dataset.cluster_map.values())
    clusters = dict((c, i) for i, c in enumerate(clusters))
    counts = [0] * len(clusters)
    for n, c in dataset.cluster_map.items():
        counts[clusters[c]] += 1
    sizes = [0] * (max(counts) + 1)
    for c in counts:
        sizes[c] += 1
    min_index = next((i for i, x in enumerate(sizes) if x), None)
    max_index = len(sizes) - next((i for i, x in enumerate(reversed(sizes)) if x), None)
    plt.bar(list(range(min_index, max_index)), sizes[min_index:max_index])
    plt.xlabel("Size of Cluster")
    plt.ylabel("Number of Clusters")
    plt.title("Size distribution of clusters")
    plt.savefig(save_dir / f"{char2name(dataset.type)}_{dataset.get_name()}_cluster_hist.png")
    plt.clf()


def whatever(
        names: List[str],
        clusters: Dict[str, str],
        distances: Optional[np.ndarray],
        similarities: Optional[np.ndarray],
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


def stats_string(count: int, split_stats: Dict[str, float]) -> str:
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
        if k != NOT_ASSIGNED:
            if (count - split_stats.get(NOT_ASSIGNED, 0)) > 0:
                output += f" {100 * v / (count - split_stats.get(NOT_ASSIGNED, 0)):>6.2f}%"
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
    elif c == "M":
        return "Molecule"
    elif c == "G":
        return "Genome"
    else:
        return "Other"
