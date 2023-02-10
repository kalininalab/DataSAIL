import os
from typing import Tuple, List, Dict, Optional

import numpy as np


def run_mash(data_folder) -> Tuple[List[str], Dict[str, str], Optional[np.ndarray], Optional[np.ndarray]]:
    cmd = f"cd mash_results && " \
          f"mash sketch -s 10000 -o ./cluster {os.path.join('..', data_folder)} && " \
          f"mash dist -t cluster.msh cluster.msh > cluster.tsv"

    if not os.path.exists("mash_results"):
        cmd = "mkdir mash_results && " + cmd

    print(cmd)
    os.system(cmd)

    names = os.listdir(data_folder)
    cluster_map = dict((n, n) for n in names)
    cluster_sim = None
    cluster_dist = read_mash_tsv("mash_results/cluster.tsv", len(names))
    cluster_names = names

    return cluster_names, cluster_map, cluster_sim, cluster_dist


def read_mash_tsv(filename, num_entities):
    output = np.zeros((num_entities, num_entities))
    with open(filename, "r") as data:
        for i, line in data.readlines()[1:]:
            for j, val in line.strip().split("\t")[1:]:
                output[i, j] = float(val)
    return output
