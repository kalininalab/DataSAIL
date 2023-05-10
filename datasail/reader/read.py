import os
from typing import Tuple, List, Optional, Callable, Dict, Any

from datasail.reader.read_genomes import read_genome_data, remove_genome_duplicates
from datasail.reader.read_molecules import read_molecule_data, remove_molecule_duplicates
from datasail.reader.read_other import read_other_data, remove_other_duplicates
from datasail.reader.read_proteins import read_protein_data, remove_protein_duplicates
from datasail.reader.utils import read_csv, DataSet, get_prefix_args


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
    e_dataset, inter = read_data_type(kwargs["e_type"])(
        kwargs["e_data"], kwargs["e_weights"], kwargs["e_sim"], kwargs["e_dist"], kwargs["e_max_sim"],
        kwargs["e_max_dist"], kwargs.get("e_id_map", None), inter, 0
    )
    e_dataset.args = kwargs["e_args"]
    f_dataset, inter = read_data_type(kwargs["f_type"])(
        kwargs["f_data"], kwargs["f_weights"], kwargs["f_sim"], kwargs["f_dist"], kwargs["f_max_sim"],
        kwargs["f_max_dist"], kwargs.get("f_id_map", None), inter, 1
    )
    f_dataset.args = kwargs["f_args"]

    return e_dataset, f_dataset, inter


def check_duplicates(**kwargs) -> Dict[str, Any]:
    """
    Remove duplicates from the input data. This is done for every input type individually by calling the respective
    function here.

    Args:
        **kwargs: Keyword arguments provided to the program

    Returns:
        The updated keyword arguments as data might have been moved
    """
    os.makedirs(os.path.join(kwargs["output"] or "", "tmp"), exist_ok=True)

    # remove duplicates from first dataset
    kwargs.update(get_remover_fun(kwargs["e_type"])("e_", kwargs["output"] or "", **get_prefix_args("e_", **kwargs)))

    # if existent, remove duplicates from second dataset as well
    if kwargs["f_type"] is not None:
        kwargs.update(get_remover_fun(kwargs["f_type"])("f_", kwargs["output"] or "", **get_prefix_args("f_", **kwargs)))

    return kwargs


def get_remover_fun(data_type: str) -> Callable:
    """
    Proxy function selecting the correct function to remove duplicates from the input data by matching the input
    data-type.

    Args:
        data_type: Input data-type

    Returns:
        A callable function to remove duplicates from an input dataset
    """
    if data_type == "P":
        return remove_protein_duplicates
    if data_type == "M":
        return remove_molecule_duplicates
    if data_type == "G":
        return remove_genome_duplicates
    return remove_other_duplicates


def read_data_type(data_type: chr) -> Callable:
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


def read_none_data(*_) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Dummy method to account for unknown data type

    Returns:
        An empty dataset according to a type of input data that cannot be read
    """
    return DataSet(), _[-2]
