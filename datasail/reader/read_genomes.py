from pathlib import Path
from typing import List, Tuple, Optional

from datasail.reader.read_molecules import remove_duplicate_values
from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT, read_folder, read_data_input
from datasail.settings import G_TYPE, UNK_LOCATION, FORM_FASTA, FORM_GENOMES


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

    def read_dir(ds: DataSet, path: Path) -> None:
        ds.data = dict(read_folder(path))
        ds.format = FORM_GENOMES

    read_data_input(data, dataset, read_dir)

    dataset = read_data(weights, strats, sim, dist, inter, index, num_clusters, tool_args, dataset)
    dataset = remove_duplicate_values(dataset, dataset.data)
    return dataset
