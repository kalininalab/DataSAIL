import os
from typing import Dict, Tuple, List

import numpy as np

from scala.cluster.wl_kernels.protein import smiles_to_grakel, pdb_to_grakel
from scala.cluster.wl_kernels.wlk import run_wl_kernel


def cluster_interactions(
        inter,
        num_drug_clusters,
        drug_cluster_map,
        num_prot_clusters,
        prot_cluster_map
) -> List[List[int]]:
    output = [[0 for _ in range(num_prot_clusters)] for _ in range(num_drug_clusters)]

    for drug, protein in inter:
        output[drug_cluster_map[drug]][prot_cluster_map[protein]] += 1

    return output


def cluster(similarity: np.ndarray, molecules: Dict[str, str], weights: Dict[str, float], **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    if isinstance(similarity, str):
        cluster_names, cluster_map, cluster_similarity, cluster_weights = clustering(molecules, similarity, **kwargs)
    elif similarity:
        cluster_names = molecules.keys()
        cluster_map = dict([(d, d) for d, _ in molecules.items()])
        cluster_similarity = similarity
        cluster_weights = weights
    else:
        cluster_names, cluster_map, cluster_similarity, cluster_weights = None, None, None, None

    return cluster_names, cluster_map, cluster_similarity, cluster_weights


def clustering(mols, cluster_method: str, **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    if cluster_method == "WLK":
        cluster_names, cluster_map, cluster_sim = run_wlk(mols, **kwargs)
    elif cluster_method == "mmseqs":
        cluster_names, cluster_map, cluster_sim = run_mmseqs(**kwargs)
    else:
        raise ValueError("Unknown clustering method.")

    cluster_weights = {}
    for key, value in cluster_map.items():
        if value not in cluster_weights:
            cluster_weights[key] = 0
        cluster_weights[key] += 1

    # cluster_map maps members to their cluster names
    return cluster_names, cluster_map, cluster_sim, cluster_weights


def run_wlk(molecules: Dict = None, **kwargs) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    if not molecules:  # cluster proteins with WLK
        graphs = [pdb_to_grakel(os.path.join(kwargs["pdb_folder"], pdb_path)) for pdb_path in
                  os.listdir(kwargs["pdb_folder"])]
        cluster_names = list(os.listdir(kwargs["pdb_folder"]))
    else:  # cluster molecules (drugs) with WLK
        cluster_names, graphs = list(zip(*((name, smiles_to_grakel(mol)) for name, mol in molecules)))

    cluster_sim = run_wl_kernel(graphs)
    cluster_map = {(name, name) for name, _ in molecules.items()}

    return cluster_names, cluster_map, cluster_sim


def run_mmseqs(**kwargs) -> Tuple[List[str], Dict[str, str], np.ndarray]:
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
    cluster_names = list(set(cluster_map.values()))

    return cluster_names, cluster_map, cluster_sim


def get_mmseqs_map(cluster_file: str) -> Dict[str, str]:
    mapping = {}
    with open(cluster_file, 'r') as f:
        for line in f.readlines():
            words = line.strip().replace('β', 'beta').split('\t')
            if len(words) != 2:
                continue
            cluster_head, cluster_member = words
            mapping[cluster_member] = cluster_member
    return mapping
