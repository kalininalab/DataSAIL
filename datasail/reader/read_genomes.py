import os
from typing import List, Tuple

from datasail.reader.utils import DataSet, read_data


def read_genome_data(
        data: str,
        weights: str,
        sim: str,
        dist: str,
        max_sim: float,
        max_dist: float,
        inter: List[Tuple[str, str]],
        index: int
) -> DataSet:
    """
    Read in genomic data, compute the weights, and distances or similarities of every entity.

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
    dataset = DataSet(type="G")
    if os.path.exists(data):
        dataset.location = data
        dataset.format = "FASTA"
    else:
        raise ValueError()

    return read_data(weights, sim, dist, max_sim, max_dist, inter, index, dataset)
