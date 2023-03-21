import logging
import pickle
from typing import Tuple, List, Dict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric

from datasail.reader.utils import DataSet


def run_ecfp(dataset: DataSet) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Compute 1024Bit-ECPFs for every molecule in the dataset and then compute pairwise Tanimoto-Scores of them.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    if dataset.type != "M":
        raise ValueError("ECFP with Tanimoto-scores can only be applied to molecular data.")

    scaffolds = {}
    logging.info("Start ECFP clustering")

    for name in dataset.names:
        scaffold = Chem.MolFromSmiles(dataset.data[name])
        if scaffold is None:
            # TODO: Report this issue
            pass
        scaffolds[name] = MakeScaffoldGeneric(scaffold)

    fps = []
    cluster_names = list(set(scaffolds.values()))
    for scaffold in cluster_names:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(scaffold, 2, nBits=1024))

    logging.info(f"Reduced {len(dataset.names)} molecules to {len(cluster_names)}")

    logging.info("Compute Tanimoto Coefficients")

    count = len(dataset.names)
    sim_matrix = np.zeros((count, count))
    for i in range(count):
        if i % 100 == 0:
            print(f"\r{i + 1} / {count}", end="")
        sim_matrix[i, i] = 1
        sim_matrix[i, :i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        sim_matrix[:i, i] = sim_matrix[i, :i]

    # pickle.dump(sim_matrix, open("/scratch/SCRATCH_SAS/roman/DataSAIL_cache/kino_lig_matrix.pkl", "wb"))

    cluster_map = dict((name, Chem.MolToSmiles(scaffolds[name])) for name in dataset.names)
    return cluster_names, cluster_map, sim_matrix
