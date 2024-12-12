from pathlib import Path
from typing import List, Callable, Generator

import numpy as np

from datasail.reader.read_genomes import read_genome_data
from datasail.reader.read_molecules import read_molecule_data
from datasail.reader.read_other import read_other_data
from datasail.reader.read_proteins import read_protein_data
from datasail.reader.utils import read_csv, DataSet
from datasail.settings import *


def read_data(**kwargs) -> Tuple[DataSet, DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Read data from the input arguments.

    Args:
        **kwargs: Arguments from commandline

    Returns:
        Two datasets storing the information on the input entities and a list of interactions between
    """
    # TODO: Semantic checks of arguments
    if kwargs[KW_INTER] is None:
        inter = None
    elif isinstance(kwargs[KW_INTER], Path):
        if kwargs[KW_INTER].is_file():
            if kwargs[KW_INTER].suffix[1:] == "tsv":
                inter = list(read_csv(kwargs[KW_INTER], sep="\t", interactions=True, stratification=True))
            elif kwargs[KW_INTER].suffix[1:] == "csv":
                inter = list(read_csv(kwargs[KW_INTER], sep=",", interactions=True, stratification=True))
            else:
                raise ValueError()
        else:
            raise ValueError()
    elif isinstance(kwargs[KW_INTER], (list, Generator)):
        inter = [(x[:2], x[2:]) for x in kwargs[KW_INTER]]
    elif isinstance(kwargs[KW_INTER], Callable):
        inter = [(x[:2], x[2:]) for x in kwargs[KW_INTER]()]
    else:
        raise ValueError()

    e_inter_strats, f_inter_strats = None, None
    if inter is not None:
        e_inter_strats, f_inter_strats = {}, {}
        for (e, f), strat in inter:
            strat = 1 - np.isnan(np.array(strat, dtype=float)).astype(int)
            if e not in e_inter_strats:
                e_inter_strats[e] = np.zeros_like(strat, dtype=int)
            if f not in f_inter_strats:
                f_inter_strats[f] = np.zeros_like(strat, dtype=int)
            # TODO: implement this in a weighted fashion
            e_inter_strats[e] |= strat
            f_inter_strats[f] |= strat


    e_dataset = read_data_type(kwargs[KW_E_TYPE])(
        kwargs[KW_E_DATA], kwargs[KW_E_WEIGHTS], kwargs[KW_E_STRAT], kwargs[KW_E_SIM], kwargs[KW_E_DIST], inter,
        e_inter_strats, 0, kwargs[KW_E_CLUSTERS], kwargs[KW_E_ARGS],
    )
    f_dataset = read_data_type(kwargs[KW_F_TYPE])(
        kwargs[KW_F_DATA], kwargs[KW_F_WEIGHTS], kwargs[KW_F_STRAT], kwargs[KW_F_SIM], kwargs[KW_F_DIST], inter,
        f_inter_strats, 1, kwargs[KW_F_CLUSTERS], kwargs[KW_F_ARGS],
    )

    return e_dataset, f_dataset, inter


def read_data_type(data_type: chr) -> Callable:
    """
    Convert single-letter representation of the type of data to handle to the full name.

    Args:
        data_type: Single letter representation of the type of data

    Returns:
        full name of the type of data
    """
    if data_type == P_TYPE:
        return read_protein_data
    elif data_type == M_TYPE:
        return read_molecule_data
    elif data_type == G_TYPE:
        return read_genome_data
    elif data_type == O_TYPE:
        return read_other_data
    else:
        return read_none_data


def read_none_data(*_) -> DataSet:
    """
    Dummy method to account for unknown data type

    Returns:
        An empty dataset according to a type of input data that cannot be read
    """
    return DataSet()  # , _[-3]
