import os
from typing import Tuple, List, Optional, Callable, Dict, Any, Generator

from datasail.reader.read_genomes import read_genome_data, remove_genome_duplicates
from datasail.reader.read_molecules import read_molecule_data, remove_molecule_duplicates
from datasail.reader.read_other import read_other_data, remove_other_duplicates
from datasail.reader.read_proteins import read_protein_data, remove_protein_duplicates
from datasail.reader.utils import read_csv, DataSet, get_prefix_args


def read_data(**kwargs) -> Tuple[DataSet, DataSet, Optional[List[Tuple[str, str]]], Optional[List[Tuple[str, str]]]]:
    """
    Read data from the input arguments.

    Args:
        **kwargs: Arguments from commandline

    Returns:
        Two datasets storing the information on the input entities and a list of interactions between
    """
    # TODO: Semantic checks of arguments
    match kwargs["inter"]:
        case None:
            old_inter = None
        case x if isinstance(x, str):
            old_inter = list(tuple(x) for x in read_csv(kwargs["inter"]))
        case x if isinstance(x, list):
            old_inter = kwargs["inter"]
        case x if isinstance(x, Callable):
            old_inter = kwargs["inter"]()
        case x if isinstance(x, Generator):
            old_inter = list(kwargs["inter"])
        case _:
            raise ValueError()

    e_dataset, inter = read_data_type(kwargs["e_type"])(
        kwargs["e_data"], kwargs["e_weights"], kwargs["e_sim"], kwargs["e_dist"], kwargs["e_max_sim"],
        kwargs["e_max_dist"], kwargs.get("e_id_map", None), old_inter, 0
    )
    e_dataset.args = kwargs["e_args"]
    f_dataset, inter = read_data_type(kwargs["f_type"])(
        kwargs["f_data"], kwargs["f_weights"], kwargs["f_sim"], kwargs["f_dist"], kwargs["f_max_sim"],
        kwargs["f_max_dist"], kwargs.get("f_id_map", None), inter, 1
    )
    f_dataset.args = kwargs["f_args"]

    if f_dataset.type is None: 
        inter = list(filter(lambda x: x[0] in e_dataset.names, inter))
    else:
        inter = list(filter(lambda x: x[0] in e_dataset.names and x[1] in f_dataset.names, inter))

    return e_dataset, f_dataset, inter, old_inter


def check_duplicates(**kwargs) -> Dict[str, Any]:
    """
    Remove duplicates from the input data. This is done for every input type individually by calling the respective
    function here.

    Args:
        **kwargs: Keyword arguments provided to the program

    Returns:
        The updated keyword arguments as data might have been moved
    """
    os.makedirs(os.path.join(kwargs.get("output", None) or "", "tmp"), exist_ok=True)

    # remove duplicates from first dataset
    kwargs.update(
        get_remover_fun(kwargs["e_type"])("e_", kwargs.get("output", None) or "", **get_prefix_args("e_", **kwargs))
    )

    # if existent, remove duplicates from second dataset as well
    if kwargs["f_type"] is not None:
        kwargs.update(
            get_remover_fun(kwargs["f_type"])("f_", kwargs.get("output", None) or "", **get_prefix_args("f_", **kwargs))
        )

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
    match data_type:
        case "P":
            return remove_protein_duplicates
        case "M":
            return remove_molecule_duplicates
        case "G":
            return remove_genome_duplicates
        case _:
            return remove_other_duplicates


def read_data_type(data_type: chr) -> Callable:
    """
    Convert single-letter representation of the type of data to handle to the full name.

    Args:
        data_type: Single letter representation of the type of data

    Returns:
        full name of the type of data
    """
    match data_type:
        case "P":
            return read_protein_data
        case "M":
            return read_molecule_data
        case "G":
            return read_genome_data
        case "O":
            return read_other_data
        case _:
            return read_none_data


def read_none_data(*_) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Dummy method to account for unknown data type

    Returns:
        An empty dataset according to a type of input data that cannot be read
    """
    return DataSet(), _[-2]
