import numpy as np
from rdkit import DataStructs

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER


def run_tanimoto(dataset: DataSet) -> None:
    """
    Compute pairwise Tanimoto-Scores of the given dataset.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for
    """
    LOGGER.info("Start Tanimoto clustering")

    if not isinstance(list(dataset.data.values())[0], np.ndarray):
        raise ValueError("Tanimoto-Clustering can only be applied to already computed embeddings.")

    count = len(dataset.cluster_names)
    dataset.cluster_similarity = np.zeros((count, count))
    fps = [dataset.data[name] for name in dataset.names]
    for i in range(count):
        dataset.cluster_similarity[i, i] = 1
        dataset.cluster_similarity[i, :i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dataset.cluster_similarity[:i, i] = dataset.cluster_similarity[i, :i]

    dataset.cluster_names = dataset.names
    dataset.cluster_map = dict((n, n) for n in dataset.names)
