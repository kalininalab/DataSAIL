import os
import shutil
from typing import Tuple, List, Dict

import numpy as np

from datasail.reader.utils import DataSet


def run_foldseek(dataset: DataSet) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    cmd = f"mkdir fs && " \
          f"cd fs && " \
          f"foldseek createdb {os.path.join('..', dataset.location)} fsdb && " \
          f"foldseek easy-search {os.path.join('..', dataset.location)} fsdb aln.m8 tmp --format-output 'query,target,fident'"

    # alternative foldseek call:
    # foldseek easy-search pdbs pdbs ali.m8 tmp (does exactly the same as above)

    print(cmd)
    os.system(cmd)

    namap = dict((n, i) for i, n in enumerate(dataset.names))

    cluster_sim = np.zeros((len(dataset.names), len(dataset.names)))
    with open("fs/aln.m8", "r") as data:
        for line in data.readlines():
            q1, q2, sim = line.strip().split("\t")[:3]
            cluster_sim[namap[q1], namap[q2]] = sim
            cluster_sim[namap[q2], namap[q1]] = sim

    shutil.rmtree("fs")

    return dataset.names, dict((n, n) for n in dataset.names), cluster_sim
