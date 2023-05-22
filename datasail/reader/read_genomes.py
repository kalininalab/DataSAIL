import os
from typing import List, Tuple, Dict, Any, Optional, Generator, Callable

from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT, read_folder


def read_genome_data(
        data: DATA_INPUT,
        weights: DATA_INPUT,
        sim: MATRIX_INPUT,
        dist: MATRIX_INPUT,
        max_sim: float,
        max_dist: float,
        id_map: Optional[str],
        inter: List[Tuple[str, str]],
        index: int
) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Read in genomic data, compute the weights, and distances or similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        max_sim: Maximal similarity between entities in two splits
        max_dist: Maximal similarity between entities in one split
        id_map: Mapping of ids in case of duplicates in the dataset
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type="G", location=None, format="FASTA")
    if isinstance(data, str):
        dataset.data = dict(read_folder(data))
        dataset.location = data
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError()

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def remove_genome_duplicates(prefix: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Remove duplicates in other data input. Currently, this is not implemented and will return the input arguments.

    Args:
        prefix: Prefix of the data. This is either 'e_' or 'f_'
        output_dir: Directory to store data to in case of detected duplicates
        **kwargs: Arguments for this data input

    Returns:
        Update arguments as teh location of the data might change and an ID-Map file might be added.
    """
    output_args = {prefix + k: v for k, v in kwargs.items()}
    return output_args
