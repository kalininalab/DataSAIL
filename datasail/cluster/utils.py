import logging
from typing import Tuple, List, Dict, Callable

import numpy as np

from datasail.reader.utils import DataSet


def cluster_param_binary_search(
        dataset: DataSet,
        init_args: Tuple,
        min_args: Tuple,
        max_args: Tuple,
        trial: Callable,
        args2str: Callable,
        gen_args: Callable,
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    cluster_names, cluster_map, cluster_sim = trial(dataset, args2str(init_args))
    num_clusters = len(cluster_names)
    logging.info(f"First round of clustering found {num_clusters} clusters.")
    if num_clusters <= 10:
        min_args = init_args
        min_clusters = num_clusters
        min_cluster_names, min_cluster_map, min_cluster_sim = cluster_names, cluster_map, cluster_sim
        max_cluster_names, max_cluster_map, max_cluster_sim = dataset.names, dict((n, n) for n in dataset.names), np.zeros((len(dataset.names), len(dataset.names)))
        max_clusters = len(max_cluster_names)
        logging.info(f"Second round of clustering found {max_clusters} clusters.")
    elif 10 < num_clusters <= 100:
        return cluster_names, cluster_map, cluster_sim
    else:
        max_args = init_args
        max_clusters = num_clusters
        max_cluster_names, max_cluster_map, max_cluster_sim = cluster_names, cluster_map, cluster_sim
        min_cluster_names, min_cluster_map, min_cluster_sim = trial(dataset, args2str(min_args))
        min_clusters = len(min_cluster_names)
        logging.info(f"First round of clustering found {min_clusters} clusters.")
    if 10 < min_clusters <= 100:
        return min_cluster_names, min_cluster_map, min_cluster_sim
    if 10 < max_clusters <= 100:
        return max_cluster_names, max_cluster_map, max_cluster_sim
    if max_clusters < 10:
        logging.warning(f"CD-HIT cannot optimally cluster the data. The maximal number of clusters is {max_clusters}.")
        return max_cluster_names, max_cluster_map, max_cluster_sim
    if 100 < min_clusters:
        logging.warning(f"CD-HIT cannot optimally cluster the data. The minimal number of clusters is {min_clusters}.")
        return min_cluster_names, min_cluster_map, min_cluster_sim

    iteration_count = 0
    while True:
        iteration_count += 1
        args = gen_args(min_args, max_args)
        cluster_names, cluster_map, cluster_sim = trial(dataset, args2str(args))
        num_clusters = len(cluster_names)
        logging.info(f"Next round of clustering ({iteration_count + 2}.) found {num_clusters} clusters.")
        if num_clusters <= 10:
            min_args = args
        elif 10 < num_clusters <= 100 or iteration_count >= 8:
            return cluster_names, cluster_map, cluster_sim
        else:
            max_args = args
