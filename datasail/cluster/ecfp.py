from typing import Tuple, List, Dict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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

    fps = []
    for name in dataset.names:
        mol = Chem.MolFromSmiles(dataset.data[name])
        if mol is None:
            # TODO: Report this issue
            fps.append(None)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    count = len(dataset.names)
    sim_matrix = np.zeros((count, count))
    for i in range(count):
        sim_matrix[i, i] = 1
        sim_matrix[i, :i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        sim_matrix[:i, i] = sim_matrix[i, :i]

    cluster_map = dict((name, name) for name in dataset.names)
    return dataset.names, cluster_map, sim_matrix
