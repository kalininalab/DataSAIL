import os
from typing import List, Tuple, Dict, Any, Optional, Generator, Callable

from datasail.reader.read_genomes import read_folder
from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT


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

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type="O", location="unknown", format="Other")
    match data:
        case str():
            if os.path.exists(data):
                dataset.data = read_folder(data)
                dataset.location = data
            else:
                raise ValueError()
        case dict():
            dataset.data = data
        case x if isinstance(x, Callable):
            dataset.data = data()
        case x if isinstance(x, Generator):
            dataset.data = dict(data)
        case _:
            raise ValueError()

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def remove_other_duplicates(prefix: str, output_dir: str, **kwargs) -> Dict[str, Any]:
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
