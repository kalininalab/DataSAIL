from typing import Tuple, List, Optional

from datasail.reader.read_genomes import read_genome_data
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_other import read_other_data
from datasail.reader.read_proteins import read_protein_data
from datasail.reader.utils import read_csv, DataSet


def read_data(**kwargs) -> Tuple[DataSet, DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Read data from the input arguments.

    Args:
        **kwargs: Arguments from commandline

    Returns:
        Two datasets storing the information on the input entities and a list of interactions between
    """
    # TODO: Semantic checks of arguments
    inter = list(tuple(x) for x in read_csv(kwargs["inter"])) if kwargs["inter"] else None
    e_dataset = read_data_type(kwargs["e_type"])(
        kwargs["e_data"], kwargs["e_weights"], kwargs["e_sim"],
        kwargs["e_dist"], kwargs["e_max_sim"], kwargs["e_max_dist"], inter, 0
    )
    e_dataset.args = kwargs["e_args"]
    f_dataset = read_data_type(kwargs["f_type"])(
        kwargs["f_data"], kwargs["f_weights"], kwargs["f_sim"],
        kwargs["f_dist"], kwargs["f_max_sim"], kwargs["f_max_dist"], inter, 1
    )
    f_dataset.args = kwargs["f_args"]
    return e_dataset, f_dataset, inter


def read_data_type(data_type: chr):
    """
    Convert single-letter representation of the type of data to handle to the full name.

    Args:
        data_type: Single letter representation of the type of data

    Returns:
        full name of the type of data
    """
    if data_type == "P":
        return read_protein_data
    if data_type == "M":
        return read_molecule_data
    if data_type == "G":
        return read_genome_data
    if data_type == "O":
        return read_other_data
    return read_none_data


def read_none_data(*_) -> DataSet:
    """
    Dummy method to account for unknown data type

    Returns:
        An empty dataset according to a type of input data that cannot be read
    """
    return DataSet()
