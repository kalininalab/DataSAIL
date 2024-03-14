import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Callable, Iterable

import h5py

from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.read_proteins import parse_fasta
from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT, read_folder, read_csv
from datasail.settings import G_TYPE, UNK_LOCATION, FORM_FASTA, FASTA_FORMATS, FORM_GENOMES


def read_genome_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        strats: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
        num_clusters: Optional[int] = None,
        tool_args: str = "",
) -> DataSet:
    """
    Read in genomic data, compute the weights, and distances or similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        strats: Stratification for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file
        num_clusters: Number of clusters to compute for this dataset
        tool_args: Additional arguments for the tool

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type=G_TYPE, location=UNK_LOCATION, format=FORM_FASTA)
    if isinstance(data, Path):
        if data.is_file():
            if data.suffix[1:].lower() in FASTA_FORMATS:
                dataset.data = parse_fasta(data)
            elif data.suffix[1:].lower() == "tsv":
                dataset.data = dict(read_csv(data, sep="\t"))
            elif data.suffix[1:].lower() == "csv":
                dataset.data = dict(read_csv(data, sep=","))
            elif data.suffix[1:].lower() == "pkl":
                with open(data, "rb") as file:
                    dataset.data = dict(pickle.load(file))
            elif data.suffix[1:].lower() == "h5":
                with h5py.File(data) as file:
                    dataset.data = dict(file[k] for k in file.keys())
            else:
                raise ValueError()
        elif data.is_dir():
            dataset.data = dict(read_folder(data))
            dataset.format = FORM_GENOMES
        else:
            raise ValueError()
        dataset.location = data
    elif (isinstance(data, list) or isinstance(data, tuple)) and isinstance(data[0], Iterable) and len(data[0]) == 2:
        dataset.data = dict(data)
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError()

    dataset = read_data(weights, strats, sim, dist, inter, index, num_clusters, tool_args, dataset)
    dataset = remove_duplicate_values(dataset, dataset.data)
    return dataset
