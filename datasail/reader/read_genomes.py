import os
from typing import List, Tuple, Dict, Any, Optional, Generator

from datasail.reader.utils import DataSet, read_data


def read_genome_data(
        data: str,
        weights: str,
        sim: str,
        dist: str,
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
    dataset = DataSet(type="G")
    if os.path.exists(data):
        dataset.data = read_folder(data)
        dataset.location = data
        dataset.format = "FASTA"
    else:
        raise ValueError()

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def read_folder(folder_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read a folder of files with arbitrary data.

    Args:
        folder_path: Path to the folder containing the data, one sample per file

    Yields:
        Pairs of sample id (filename) and the absolute path to that file
    """
    for filename in os.listdir(folder_path):
        yield ".".join(filename.split(".")[:-1]), os.path.abspath(os.path.join(folder_path, filename))


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
