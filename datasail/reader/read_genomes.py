from typing import List, Tuple, Optional, Generator, Callable

from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT, read_folder
from datasail.settings import G_TYPE, UNK_LOCATION, FORM_FASTA


def read_genome_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        max_sim: float = 1.0,
        max_dist: float = 1.0,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
        tool_args: str = "",
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
        tool_args: Additional arguments for the tool

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type=G_TYPE, location=UNK_LOCATION, format=FORM_FASTA)
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

    dataset = read_data(weights, sim, dist, max_sim, max_dist, inter, index, tool_args, dataset)
    dataset = remove_duplicate_values(dataset, dataset.data)
    return dataset
