import os
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from rdkit.Chem import MolFromSmiles
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

from scala.cluster.wl_kernels.protein import mol_to_grakel, pdb_to_grakel
from scala.cluster.wl_kernels.wlk import run_wl_kernel


def cluster_interactions(
        inter,
        drug_cluster_map: Dict[str, Union[str, int]],
        drug_cluster_names: List[Union[str, int]],
        prot_cluster_map: Dict[str, Union[str, int]],
        prot_cluster_names: List[Union[str, int]],
) -> np.ndarray:
    drug_mapping = dict((y, x) for x, y in enumerate(drug_cluster_names))
    prot_mapping = dict((y, x) for x, y in enumerate(prot_cluster_names))

    output = np.zeros((len(drug_cluster_names), len(prot_cluster_names)))
    for drug, protein in inter:
        output[drug_mapping[drug_cluster_map[drug]]][prot_mapping[prot_cluster_map[protein]]] += 1
        output[prot_mapping[prot_cluster_map[protein]]][drug_mapping[drug_cluster_map[drug]]] += 1

    return output


def cluster(
        similarity: Optional[Union[np.ndarray, str]],
        distance: Optional[Union[np.ndarray, str]],
        molecules: Optional[Dict[str, str]],
        weights: Dict[str, float],
        **kwargs
) -> Tuple[List[Union[str, int]], Dict[str, str], Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    cluster_similarity, cluster_distance = None, None
    if isinstance(similarity, str):
        cluster_names, cluster_map, cluster_similarity, cluster_weights = \
            similarity_clustering(molecules, similarity, **kwargs)
    if isinstance(distance, str):
        cluster_names, cluster_map, cluster_distance, cluster_weights = \
            distance_clustering(molecules, distance, **kwargs)
    elif similarity is not None or distance is not None:
        cluster_names = list(molecules.keys())
        cluster_map = dict([(d, d) for d, _ in molecules.items()])
        cluster_similarity = similarity
        cluster_distance = distance
        cluster_weights = weights
    else:
        cluster_names, cluster_map, cluster_weights = None, None, None

    while len(cluster_names) > 50:
        cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights = additional_clustering(
            cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights
        )

    return cluster_names, cluster_map, cluster_similarity, cluster_distance, cluster_weights


def similarity_clustering(mols: Optional, cluster_method: str, **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    if cluster_method.lower() == "wlk":
        cluster_names, cluster_map, cluster_sim = run_wlk(mols, **kwargs)
    elif cluster_method.lower() == "mmseqs":
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


def distance_clustering(mols: Optional, cluster_method: str, **kwargs) -> Tuple[
    List[str], Dict[str, str], np.ndarray, Dict[str, float],
]:
    return [], {}, np.array(1), {}


def additional_clustering(
        cluster_names: List[Union[str, int]],
        cluster_map: Dict[str, str],
        cluster_similarity: Optional[np.ndarray],
        cluster_distance: Optional[np.ndarray],
        cluster_weights: Dict[str, float],
):
    # assert (cluster_similarity is None) != (cluster_distance is None)
    if cluster_similarity is not None:
        ca = AffinityPropagation(affinity='precomputed', random_state=42)
        cluster_matrix = np.array(cluster_similarity, dtype=float)
    else:
        ca = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=np.average(cluster_distance) * 0.9
        )
        cluster_matrix = np.array(cluster_distance, dtype=float)

    labels = ca.fit_predict(cluster_matrix)

    old_cluster_map = dict((y, x) for x, y in enumerate(cluster_names))
    new_cluster_names = list(np.unique(labels))
    new_cluster_map = dict((n, labels[old_cluster_map[c]]) for n, c in cluster_map.items())

    new_cluster_matrix = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    cluster_count = np.zeros((len(new_cluster_names), len(new_cluster_names)))
    for i in range(len(cluster_names)):
        for j in range(i + 1, len(cluster_names)):
            if labels[i] != labels[j]:
                new_cluster_matrix[labels[i], labels[j]] += cluster_matrix[i, j]
                cluster_count[labels[i], labels[j]] += 1

                new_cluster_matrix[labels[j], labels[i]] += cluster_matrix[i, j]
                cluster_count[labels[j], labels[i]] += 1
    new_cluster_matrix /= (cluster_count + np.eye(max(labels) + 1))

    new_cluster_weights = {}
    for name in cluster_names:
        new_cluster = new_cluster_map[name]
        if new_cluster not in new_cluster_weights:
            new_cluster_weights[new_cluster] = 0
        new_cluster_weights[new_cluster] += cluster_weights[name]

    if cluster_similarity is not None:
        return new_cluster_names, new_cluster_map, new_cluster_matrix, None, new_cluster_weights
    return new_cluster_names, new_cluster_map, None, new_cluster_matrix, new_cluster_weights


def run_wlk(molecules: Dict, **kwargs) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    if os.path.isfile(list(molecules.values())[1]):  # cluster proteins with WLK
        cluster_names, graphs = list(zip(*((name, pdb_to_grakel(pdb_path)) for name, pdb_path in molecules.items())))
    else:  # cluster molecules (drugs) with WLK
        cluster_names, graphs = list(zip(*((name, mol_to_grakel(notation_to_mol(mol))) for name, mol in molecules.items())))

    cluster_sim = run_wl_kernel(graphs)
    cluster_map = dict((name, name) for name, _ in molecules.items())

    return cluster_names, cluster_map, cluster_sim


def notation_to_mol(mol: str):
    return MolFromSmiles(mol)


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
