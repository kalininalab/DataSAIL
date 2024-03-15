import copy
from typing import Literal

import numpy as np
from rdkit import DataStructs

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER

SIM_OPTIONS = Literal[
    "AllBit", "Asymmetric", "BraunBlanquet", "Cosine", "Dice", "Kulczynski", "McConnaughey", "OnBit", "RogotGoldberg",
    "Russel", "Sokal", "Tanimoto", "Jaccard"
]


def get_rdkit_fct(method: SIM_OPTIONS):
    """
    Get the RDKit function for the given similarity measure.

    Args:
        method: The name of the similarity measure to get the function for.

    Returns:
        The RDKit function for the given similarity measure.
    """
    if method == "AllBit":
        return DataStructs.BulkAllBitSimilarity
    if method == "Asymmetric":
        return DataStructs.BulkAsymmetricSimilarity
    if method == "BraunBlanquet":
        return DataStructs.BulkBraunBlanquetSimilarity
    if method == "Cosine":
        return DataStructs.BulkCosineSimilarity
    if method == "Dice":
        return DataStructs.BulkDiceSimilarity
    if method == "Kulczynski":
        return DataStructs.BulkKulczynskiSimilarity
    if method == "McConnaughey":
        return DataStructs.BulkMcConnaugheySimilarity
    if method == "OnBit":
        return DataStructs.BulkOnBitSimilarity
    if method == "RogotGoldberg":
        return DataStructs.BulkRogotGoldbergSimilarity
    if method == "Russel":
        return DataStructs.BulkRusselSimilarity
    if method == "Sokal":
        return DataStructs.BulkSokalSimilarity
    if method == "Tanimoto" or method == "Jaccard":
        return DataStructs.BulkTanimotoSimilarity
    if method == "Tversky":
        return DataStructs.BulkTverskySimilarity
    raise ValueError(f"Unknown method {method}")


def iterable2intvect(it):
    """
    Convert an iterable to an RDKit LongSparseIntVect.

    Args:
        it: The iterable to convert.

    Returns:
        The RDKit LongSparseIntVect.
    """
    output = DataStructs.LongSparseIntVect(len(it))
    for i, v in enumerate(it):
        output[i] = max(-2_147_483_648, min(2_147_483_647, int(v)))
    return output


def iterable2bitvect(it):
    """
    Convert an iterable to an RDKit ExplicitBitVect.

    Args:
        it: The iterable to convert.

    Returns:
        The RDKit ExplicitBitVect.
    """
    output = DataStructs.ExplicitBitVect(len(it))
    output.SetBitsFromList([i for i, v in enumerate(it) if v])
    return output


def run_tanimoto(dataset: DataSet, method: SIM_OPTIONS = "Tanimoto") -> None:
    """
    Compute pairwise Tanimoto-Scores of the given dataset.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for
        method: The similarity measure to use. Default is "Tanimoto".
    """
    LOGGER.info("Start Tanimoto clustering")

    embed = dataset.data[dataset.names[0]]
    if isinstance(embed, (list, tuple, np.ndarray)):
        if isinstance(embed[0], int) or np.issubdtype(embed[0].dtype, int):
            if method in ["AllBit", "Asymmetric", "BraunBlanquet", "Cosine", "Kulczynski", "McConnaughey", "OnBit",
                          "RogotGoldberg", "Russel", "Sokal"]:
                dataset.data = {k: iterable2bitvect(v) for k, v in dataset.data.items()}
            else:
                dataset.data = {k: iterable2intvect(v) for k, v in dataset.data.items()}
            embed = dataset.data[dataset.names[0]]
        else:
            raise ValueError("Embeddings with non-integer elements are not supported at the moment.")
    if not isinstance(embed,
                      (DataStructs.ExplicitBitVect, DataStructs.LongSparseIntVect, DataStructs.IntSparseIntVect)):
        raise ValueError(f"Unsupported embedding type {type(embed)}. Please use either RDKit datastructures, lists, "
                         f"tuples or one-dimensional numpy arrays.")
    fps = [dataset.data[name] for name in dataset.names]
    run(dataset, fps, method)

    dataset.cluster_names = copy.deepcopy(dataset.names)
    dataset.cluster_map = dict((n, n) for n in dataset.names)


def run(dataset, fps, method):
    """
    Compute pairwise similarities of the given fingerprints.

    Args:
        dataset: The dataset to compute pairwise similarities for.
        fps: The fingerprints to compute pairwise similarities for.
        method: The similarity measure to use.
    """
    fct = get_rdkit_fct(method)
    dataset.cluster_similarity = np.zeros((len(fps), len(fps)))
    for i in range(len(fps)):
        dataset.cluster_similarity[i, i] = 1
        dataset.cluster_similarity[i, :i] = fct(fps[i], fps[:i])
        dataset.cluster_similarity[:i, i] = dataset.cluster_similarity[i, :i]

    min_val = np.min(dataset.cluster_similarity)
    max_val = np.max(dataset.cluster_similarity)
    dataset.cluster_similarity = (dataset.cluster_similarity - min_val) / (max_val - min_val)
