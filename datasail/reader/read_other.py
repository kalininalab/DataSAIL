import os

from datasail.reader.utils import DataSet, read_data


def read_other_data(data, weights, sim, dist, max_sim, max_dist, inter, index) -> DataSet:
    """
    Read in other data, i.e., non-protein, non-molecular, and non-genomic data, compute the weights, and distances or
    similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        max_sim: Maximal similarity between entities in two splits
        max_dist: Maximal similarity between entities in one split
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type="O")
    if os.path.exists(data):
        dataset.location = data
    else:
        raise ValueError()

    return read_data(weights, sim, dist, max_sim, max_dist, inter, index, dataset)

