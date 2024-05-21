import copy
from typing import Literal, get_args, Union, Callable, Any

import numpy as np
import scipy
from rdkit import DataStructs

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER

SIM_OPTIONS = Literal[
    "allbit", "asymmetric", "braunblanquet", "cosine", "dice", "kulczynski", "onbit", "rogotgoldberg",
    "russel", "sokal", "tanimoto"
]

# unbounded: chebyshev, cityblock, euclidean, mahalanobis, manhattan, mcconnaughey, minkowski, sqeuclidean
# produces inf or nan: correlation, cosine, jensenshannon, seuclidean, braycurtis
# boolean only: dice, kulczynski1, russelrao, sokalsneath
# matching == hamming, manhattan == cityblock (inofficial)
DIST_OPTIONS = Literal[
    "canberra", "hamming", "jaccard", "matching", "rogerstanimoto", "sokalmichener", "yule"
]


def get_rdkit_fct(method: SIM_OPTIONS) -> Callable[[Any, Any], np.ndarray]:
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
    if method == "onbit":
        return DataStructs.BulkOnBitSimilarity
    if method == "rogotgoldberg":
        return DataStructs.BulkRogotGoldbergSimilarity
    if method == "russel":
        return DataStructs.BulkRusselSimilarity
    if method == "tanimoto":
        return DataStructs.BulkTanimotoSimilarity
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


def iterable2intvect(it) -> DataStructs.IntSparseIntVect:
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


def iterable2bitvect(it) -> DataStructs.ExplicitBitVect:
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
                if method in ["allbit", "asymmetric", "braunblanquet", "cosine", "kulczynski", "onbit",
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
        dtype = np.bool_ if ["jaccard", "rogerstanimoto", "sokalmichener", "yule"] else np.float64
        if isinstance(embed, (
                list, tuple, DataStructs.ExplicitBitVect, DataStructs.LongSparseIntVect, DataStructs.IntSparseIntVect)):
            dataset.data = {k: np.array(list(v), dtype=dtype) for k, v in dataset.data.items()}
        if not isinstance(dataset.data[dataset.names[0]], np.ndarray):
            raise ValueError(
                f"Unsupported embedding type {type(embed)}. Please use either RDKit datastructures, lists, "
                f"tuples or one-dimensional numpy arrays.")
    else:
        raise ValueError(f"Unknown method {method}")
    fps = [dataset.data[name] for name in dataset.names]

    run(dataset, fps, method)
    dataset.cluster_names = copy.deepcopy(dataset.names)
    dataset.cluster_map = {n: n for n in dataset.names}


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
        dataset.cluster_similarity = rdkit_sim(fps, method)
        if method == "mcconnaughey":
            dataset.cluster_similarity = dataset.cluster_similarity + 1 / 2
    elif method in get_args(DIST_OPTIONS):
        if method == "mahalanobis" and len(fps) <= len(fps[0]):
            raise ValueError(
                f"For clustering with the Mahalanobis method, you have to have more observations that dimensions in "
                f"the embeddings. The number of samples ({len(fps)}) is too small; the covariance matrix is singular. "
                f"For observations with {len(fps[0])} dimensions, at least {len(fps[0]) + 1} observations are required."
            )
        dataset.cluster_distance = scipy.spatial.distance.cdist(fps, fps, metric=method)
        if method == "canberra":
            dataset.cluster_distance = dataset.cluster_distance / len(fps[0])
        elif method == "yule":
            dataset.cluster_distance /= 2


if __name__ == '__main__':
    data = np.array([
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ], dtype=np.bool_)
    for dist in ["rogerstanimoto", "sokalmichener", "yule"]:
        run_vector(ds := DataSet(data={chr(97 + i): v for i, v in enumerate(data)}, names=[chr(97 + i) for  i in range(len(data))]), dist)
        print(f"{dist}\t{np.min(ds.cluster_distance)}\t{np.max(ds.cluster_distance)}", ds.cluster_distance, sep="\n", end="\n\n")
