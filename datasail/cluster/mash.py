import logging
import os
import shutil
from typing import Tuple, List, Dict, Optional

import numpy as np

from datasail.reader.utils import DataSet


def run_mash(dataset: DataSet) -> Tuple[List[str], Dict[str, str], Optional[np.ndarray]]:
    cmd = f"mkdir mash_results && " \
          f"cd mash_results && " \
          f"mash sketch -s 10000 -o ./cluster {os.path.join('..', dataset.location, '*.fna')} && " \
          f"mash dist -t cluster.msh cluster.msh > cluster.tsv"

    if os.path.exists("mash_results"):
        cmd = "rm -rf mash_results && " + cmd

    logging.info("Start MASH clustering")

    os.system(cmd)

    names = dataset.names
    cluster_map = dict((n, n) for n in names)
    cluster_dist = read_mash_tsv("mash_results/cluster.tsv", len(names))
    cluster_names = names

    shutil.rmtree("mash_results")

    return cluster_names, cluster_map, cluster_dist


def read_mash_tsv(filename: str, num_entities: int) -> np.ndarray:
    """
    Read in the TSV file with pairwise distances produces by MASH.

    Args:
        filename: Filename of the file to read from
        num_entities: Number of entities in the set

    Returns:
        Symmetric 2D-numpy array storing pairwise distances
    """
    output = np.zeros((num_entities, num_entities))
    with open(filename, "r") as data:
        for i, line in enumerate(data.readlines()[1:]):
            for j, val in enumerate(line.strip().split("\t")[1:]):
                output[i, j] = float(val)
    return output
