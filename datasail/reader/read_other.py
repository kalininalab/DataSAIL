import os
from typing import List, Tuple, Optional, Generator, Callable

from datasail.reader.read_genomes import read_folder
from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT
from datasail.settings import O_TYPE, UNK_LOCATION, FORM_OTHER


def read_other_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        max_sim: float = 1.0,
        max_dist: float = 1.0,
        id_map: Optional[str] = None,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
        tool_args: str = "",
) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
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
        id_map: Mapping of ids in case of duplicates in the dataset
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file
        tool_args: Additional arguments for the tool

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type=O_TYPE, location=UNK_LOCATION, format=FORM_OTHER)
    if isinstance(data, str):
        if os.path.exists(data):
            dataset.data = read_folder(data)
            dataset.location = data
        else:
            raise ValueError()
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError()

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, inter, index, tool_args, dataset)
    dataset = remove_duplicate_values(dataset, dataset.data)

    return dataset, inter
