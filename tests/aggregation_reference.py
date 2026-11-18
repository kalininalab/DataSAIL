"""Frozen pre-scalability aggregation from de5e5a5; keep reduction semantics unchanged."""
from collections import defaultdict
from typing import List, Literal, Union

import numpy as np

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER


def labels2clusters(
        labels: Union[List, np.ndarray],
        dataset: DataSet,
        cluster_matrix: np.ndarray,
        linkage: Literal["average", "single", "complete"],
) -> DataSet:
    """
    Convert a list of labels to a clustering and insert it into the dataset. This also updates cluster_weights and
    distance or similarity metrics.

    Args:
        labels: List of nwe cluster labels in the order of dataset.cluster_names
        dataset: The dataset that is clustered
        cluster_matrix: Matrix storing distance or similarity values
        linkage: List of linkage methods to use for the clustering

    Returns:
        The updated dataset and the converged-flag
    """
    # extract the names of the new clusters and compute a mapping from the element names to the clusters
    old_cluster_map = dict((y, x) for x, y in enumerate(dataset.cluster_names))
    new_cluster_names = list(np.unique(labels))
    new_cluster_map = dict((n, labels[old_cluster_map[c]]) for n, c in dataset.cluster_map.items())

    # compute the distance or similarity matrix for the new clusters as the average sim/dist between their members
    # new_cluster_matrix = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    new_cluster_matrix = [[[] for _ in new_cluster_names] for _ in new_cluster_names]
    for i in range(len(dataset.cluster_names)):
        new_cluster_matrix[labels[i]][labels[i]].append(cluster_matrix[i, i])
        for j in range(i + 1, len(dataset.cluster_names)):
            if labels[i] != labels[j]:
                new_cluster_matrix[labels[i]][labels[j]].append(cluster_matrix[i, j])
                new_cluster_matrix[labels[j]][labels[i]].append(cluster_matrix[j, i])

    links = {"average": np.mean, "single": np.min, "complete": np.max}
    for i in range(len(new_cluster_matrix)):
        for j in range(len(new_cluster_matrix[i])):
            if len(new_cluster_matrix[i][j]) > 0:
                new_cluster_matrix[i][j] = links[linkage](new_cluster_matrix[i][j])
            else:
                new_cluster_matrix[i][j] = 0
    new_cluster_matrix = np.array(new_cluster_matrix)

    # compute the mapping of new clusters to their weights as the sum of their members weights
    new_cluster_weights = {}
    for i, name in enumerate(dataset.cluster_names):
        new_cluster = labels[i]
        if new_cluster not in new_cluster_weights:
            new_cluster_weights[new_cluster] = 0
        new_cluster_weights[new_cluster] += dataset.cluster_weights[name]
    if dataset.cluster_stratification is not None:
        new_cluster_stratification = defaultdict(lambda: np.zeros(len(dataset.classes)))
        for i, name in enumerate(dataset.cluster_names):
            new_cluster = labels[i]
            new_cluster_stratification[new_cluster] += dataset.cluster_stratification[name]
    else:
        new_cluster_stratification = None
    if dataset.cluster_interaction_stratification is not None:
        new_cluster_interaction_stratification = defaultdict(lambda: np.zeros(len(dataset.interaction_classes)))
        for i, name in enumerate(dataset.cluster_names):
            new_cluster = labels[i]
            new_cluster_interaction_stratification[new_cluster] += dataset.cluster_interaction_stratification[name]
    else:
        new_cluster_interaction_stratification = None
    LOGGER.info(f"Reduced number of clusters to {len(new_cluster_names)}.")

    dataset.cluster_names = new_cluster_names
    dataset.cluster_map = new_cluster_map
    dataset.cluster_weights = new_cluster_weights
    dataset.cluster_stratification = new_cluster_stratification
    dataset.cluster_interaction_stratification = new_cluster_interaction_stratification

    # store the matrix at the correct field and set the main diagonal to either 1 or 0 depending on dist or sim
    if dataset.cluster_similarity is not None:
        dataset.cluster_similarity = np.maximum(new_cluster_matrix, np.eye(len(new_cluster_matrix)))
    else:
        dataset.cluster_distance = np.minimum(new_cluster_matrix, 1 - np.eye(len(new_cluster_matrix)))

    return dataset
