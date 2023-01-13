import os
from typing import Dict

import numpy as np

from scala.cluster.wl_kernels.protein import smiles_to_grakel, pdb_to_grakel
from scala.cluster.wl_kernels.wlk import run_wl_kernel


def cluster(mols, cluster_method: str, **kwargs):
    if cluster_method == "WLK":
        cluster_map, cluster_sim = run_wl_kernel(mols)
    elif cluster_method == "mmseqs":
        cluster_map, cluster_sim = run_mmseqs(**kwargs)
    else:
        raise ValueError("Unknown clustering method.")

    # cluster_map maps members to their cluster names
    return cluster_map, cluster_sim


def run_wlk(molecules: Dict = None, **kwargs):
    if not molecules:  # cluster proteins with WLK
        graphs = [pdb_to_grakel(pdb_path) for pdb_path in kwargs["pdb_folder"]]
    else:
        graphs = [smiles_to_grakel(mol) for _, mol in molecules]

    cluster_sim = run_wl_kernel(graphs)
    cluster_map = {(name, name) for name, _ in molecules}

    return cluster_map, cluster_sim


def run_mmseqs(**kwargs):
    cmd = f"mmseqs " \
          f"easy-linclust " \
          f"{kwargs['input']} " \
          f"mmseqs_out " \
          f"mmseqs_tmp " \
          f"--similarity-type 2 " \
          f"--cov-mode 0 " \
          f"-c 1.0 " \
          f"--min-seq-id 0.0"
    print(cmd)
    os.system(cmd)

    cluster_map = get_mmseqs_map("mmseqs_out_cluster.tsv")
    cluster_sim = np.zeros((len(cluster_map), len(cluster_map)))

    return cluster_map, cluster_sim


def get_mmseqs_map(cluster_file):
    mapping = {}
    with open(cluster_file, 'r') as f:
        for line in f.readlines():
            words = line.strip().replace('β', 'beta').split('\t')
            if len(words) != 2:
                continue
            cluster_head, cluster_member = words
            mapping[cluster_member] = cluster_member
    return mapping
