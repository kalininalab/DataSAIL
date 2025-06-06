from pathlib import Path
from typing import Optional

from datasail.dataset import DataSet
from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.utils import read_data, read_folder, read_input_data
from datasail.constants import G_TYPE, UNK_LOCATION, FORM_FASTA, FORM_GENOMES, DATA_INPUT, MATRIX_INPUT


def read_genome_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        strats: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        inter: Optional[list[tuple]] = None,
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

    def read_dir(ds: DataSet, path: Path) -> None:
        ds.data = dict(read_folder(path))
        ds.format = FORM_GENOMES

    read_input_data(data, dataset, read_dir)

    dataset = read_data(weights, strats, sim, dist, inter, index, num_clusters, tool_args, dataset)
    dataset = remove_duplicate_values(dataset, dataset.data)
    return dataset
