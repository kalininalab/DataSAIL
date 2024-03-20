import copy
from typing import Literal, get_args, Union

import numpy as np
import scipy
from rdkit import DataStructs

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER

SIM_OPTIONS = Literal[
    "allbit", "asymmetric", "braunblanquet", "cosine", "dice", "kulczynski", "mcconnaughey", "onbit", "rogotgoldberg",
    "russel", "sokal"
]

# produces inf or nan: correlation, cosine, jensenshannon, seuclidean, braycurtis
# boolean only: dice, kulczynski1, rogerstanimoto, russelrao, sokalmichener, sokalsneath, yule
# matching == hamming, manhattan == cityblock (inofficial)
DIST_OPTIONS = Literal[
    "canberra", "chebyshev", "cityblock", "euclidean", "hamming", "jaccard", "mahalanobis", "manhattan", "matching",
    "minkowski", "sqeuclidean", "tanimoto"
]


def get_rdkit_fct(method: SIM_OPTIONS):
    """
    Get the RDKit function for the given similarity measure.

    Args:
        method: The name of the similarity measure to get the function for.

    Returns:
        The RDKit function for the given similarity measure.
    """
    if method == "allbit":
        return DataStructs.BulkAllBitSimilarity
    if method == "asymmetric":
        return DataStructs.BulkAsymmetricSimilarity
    if method == "braunblanquet":
        return DataStructs.BulkBraunBlanquetSimilarity
    if method == "cosine":
        return DataStructs.BulkCosineSimilarity
    if method == "dice":
        return DataStructs.BulkDiceSimilarity
    if method == "kulczynski":
        return DataStructs.BulkKulczynskiSimilarity
    if method == "mcconnaughey":
        return DataStructs.BulkMcConnaugheySimilarity
    if method == "onbit":
        return DataStructs.BulkOnBitSimilarity
    if method == "rogotgoldberg":
        return DataStructs.BulkRogotGoldbergSimilarity
    if method == "russel":
        return DataStructs.BulkRusselSimilarity
    if method == "sokal":
        return DataStructs.BulkSokalSimilarity
    raise ValueError(f"Unknown method {method}")


def rdkit_sim(fps, method: SIM_OPTIONS) -> np.ndarray:
    """
    Compute the similarity between elements of a list of rdkit vectors.

    Args:
        fps: List of RDKit vectors to fastly compute the similarity matrix
        method: Name of the method to use for calculation

    Returns:

    """
    fct = get_rdkit_fct(method)
    matrix = np.zeros((len(fps), len(fps)))
    for i in range(len(fps)):
        matrix[i, i] = 1
        matrix[i, :i] = fct(fps[i], fps[:i])
        matrix[:i, i] = matrix[i, :i]
    return matrix


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


def run_vector(dataset: DataSet, method: SIM_OPTIONS = "tanimoto") -> None:
    """
    Compute pairwise Tanimoto-Scores of the given dataset.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for
        method: The similarity measure to use. Default is "tanimoto".
    """
    LOGGER.info("Start Tanimoto clustering")
    method = method.lower()

    embed = dataset.data[dataset.names[0]]
    if method in get_args(SIM_OPTIONS):
        if isinstance(embed, (list, tuple, np.ndarray)):
            if isinstance(embed[0], int) or np.issubdtype(embed[0].dtype, int):
                if method in ["allbit", "asymmetric", "braunblanquet", "cosine", "kulczynski", "mcconnaughey", "onbit",
                              "rogotgoldberg", "russel", "sokal"]:
                    dataset.data = {k: iterable2bitvect(v) for k, v in dataset.data.items()}
                else:
                    dataset.data = {k: iterable2intvect(v) for k, v in dataset.data.items()}
                embed = dataset.data[dataset.names[0]]
            else:
                raise ValueError(f"Embeddings with non-integer elements are not supported for {method}.")
        if not isinstance(embed, (
                DataStructs.ExplicitBitVect, DataStructs.LongSparseIntVect, DataStructs.IntSparseIntVect
        )):
            raise ValueError(
                f"Unsupported embedding type {type(embed)}. Please use either RDKit datastructures, lists, "
                f"tuples or one-dimensional numpy arrays.")
    elif method in get_args(DIST_OPTIONS):
        if isinstance(embed, (list, tuple, DataStructs.ExplicitBitVect, DataStructs.LongSparseIntVect, DataStructs.IntSparseIntVect)):
            dataset.data = {k: np.array(list(v), dtype=np.float64) for k, v in dataset.data.items()}
        if not isinstance(dataset.data[dataset.names[0]], np.ndarray):
            raise ValueError(
                f"Unsupported embedding type {type(embed)}. Please use either RDKit datastructures, lists, "
                f"tuples or one-dimensional numpy arrays.")
    else:
        raise ValueError(f"Unknown method {method}")
    fps = [dataset.data[name] for name in dataset.names]
    run(dataset, fps, method)

    dataset.cluster_names = copy.deepcopy(dataset.names)
    dataset.cluster_map = dict((n, n) for n in dataset.names)


def scale_min_max(matrix: np.ndarray) -> np.ndarray:
    """
    Transform features by scaling each feature to the 0-1 range.

    Args:
        matrix: The numpy array to be scaled

    Returns:
        The scaled numpy array
    """
    min_val, max_val = np.min(matrix), np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


def run(
        dataset: DataSet,
        fps: Union[np.ndarray, DataStructs.ExplicitBitVect, DataStructs.LongSparseIntVect,
            DataStructs.IntSparseIntVect],
        method: Union[SIM_OPTIONS, DIST_OPTIONS],
) -> None:
    """
    Compute pairwise similarities of the given fingerprints.

    Args:
        dataset: The dataset to compute pairwise similarities for.
        fps: The fingerprints to compute pairwise similarities for.
        method: The similarity measure to use.
    """
    if method in get_args(SIM_OPTIONS):
        dataset.cluster_similarity = scale_min_max(rdkit_sim(fps, method))
    elif method in get_args(DIST_OPTIONS):
        if method == "mahalanobis" and len(fps) <= len(fps[0]):
            raise ValueError(
                f"For clustering with the Mahalanobis method, you have to have more observations that dimensions in "
                f"the embeddings. The number of samples ({len(fps)}) is too small; the covariance matrix is singular. "
                f"For observations with {len(fps[0])} dimensions, at least {len(fps[0]) + 1} observations are required."
            )
        dataset.cluster_distance = scale_min_max(scipy.spatial.distance.cdist(
            fps, fps, metric={"manhattan": "cityblock", "tanimoto": "jaccard"}.get(method, method)
        ))
