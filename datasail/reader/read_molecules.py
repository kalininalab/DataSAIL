import os

import numpy as np

from datasail.reader.utils import read_csv, read_similarity_file, count_inter, DataSet


def read_molecule_data(data, weights, sim, dist, max_sim, max_dist, inter, index) -> DataSet:
    """
    Parse molecular data into a dataset.

    Args:
        data: Location where the actual molecular data is stored
        weights: weights of the molecules in the entity
        sim: similarity metric between pairs of molecules
        dist: distance metrix between pairs of molecules
        max_sim: maximal similarity of pairs of molecules when splitting
        max_dist: maximal distance of pairs of molecules when splitting
        inter: interaction of the molecules and another entity
        index: position of the molecules in the interactions, either 0 or 1

    Returns:
        molecules parsed into a dataset
    """
    dataset = DataSet(type="M")
    if data.lower().endswith(".tsv"):
        dataset.data = dict(read_csv(data, False, "\t"))
    elif os.path.isdir(data):
        dataset.location = data
    else:
        raise ValueError()

    # parse molecular weights
    if weights is not None:
        dataset.weights = dict((x, float(v)) for x, v in read_csv(weights, False, "\t"))
    elif inter is not None:
        dataset.weights = dict(count_inter(inter, index))
    else:
        dataset.weights = dict((d, 1) for d in list(dataset.data.keys()))

    # parse molecular similarity
    if sim is None and dist is None:
        dataset.similarity = np.ones((len(dataset.data), len(dataset.data)))
        dataset.names = list(dataset.data.keys())
        dataset.threshold = 1
    elif sim is not None and os.path.isfile(sim):
        dataset.names, dataset.similarity = read_similarity_file(sim)
        dataset.threshold = max_sim
    elif dist is not None and os.path.isfile(dist):
        dataset.names, dataset.distance = read_similarity_file(dist)
        dataset.threshold = max_dist
    else:
        if sim is not None:
            dataset.similarity = sim
            dataset.threshold = max_sim
        else:
            dataset.distance = dist
            dataset.threshold = max_dist
        dataset.names = list(dataset.data.keys())

    return dataset
