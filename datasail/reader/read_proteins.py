import os
from typing import Generator, Tuple, Dict, List, Any, Optional

from datasail.reader.utils import read_csv, DataSet, read_data


def read_protein_data(
        data: str,
        weights: str,
        sim: str,
        dist: str,
        max_sim: float,
        max_dist: float,
        id_map: Optional[str],
        inter: List[Tuple[str, str]],
        index: int,
) -> Tuple[DataSet, Optional[List[Tuple[str, str]]]]:
    """
    Read in protein data, compute the weights, and distances or similarities of every entity.

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
    dataset = DataSet(type="P")
    if data.split(".")[-1].lower() in {"fasta", "fa", "fna"}:
        dataset.data = parse_fasta(data)
        dataset.format = "FASTA"
    elif os.path.isfile(data):
        dataset.data = dict(read_csv(data))
        dataset.format = "FASTA"
    elif os.path.isdir(data):
        dataset.data = dict(read_folder(data, ".pdb"))
        dataset.format = "PDB"
    else:
        raise ValueError()
    dataset.location = data

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def remove_protein_duplicates(prefix: str, **kwargs) -> Dict[str, Any]:
    """
    Remove duplicates in protein input. This is done for FASTA input as well as for PDB input.

    Args:
       prefix: Prefix of the data. This is either 'e_' or 'f_'
        **kwargs: Arguments for this data input

    Returns:
        Update arguments as teh location of the data might change and an ID-Map file might be added.
    """
    # read the data
    output_args = {prefix + k: v for k, v in kwargs.items()}
    if kwargs["data"].split(".")[-1].lower() in {"fasta", "fa", "fna"}:
        sequences = parse_fasta(kwargs["data"])
    elif os.path.isfile(kwargs["data"]):
        sequences = dict(read_csv(kwargs["data"]))
    else:
        # input is PDB data. TODO: Identity detection with PDB files
        return output_args

    id_list = []  # unique ids
    id_map = {}  # mapping of all ids to their representative
    duplicate_found = False
    for idx, seq in sequences.items():
        for q_id in id_list:
            if seq == sequences[q_id]:
                id_map[idx] = q_id
                duplicate_found = True
        if idx not in id_map:
            id_list.append(idx)
            id_map[idx] = idx

    # no duplicates found, no further action necessary
    if not duplicate_found:
        return output_args

    # store the new FASTA file
    fasta_filename = os.path.abspath(os.path.join(kwargs["output"], "tmp", prefix + "seqs.fasta"))
    with open(fasta_filename, "w") as out:
        for idx in id_list:
            print(f">{idx}\n{sequences[idx]}", file=out)
    output_args[prefix + "data"] = fasta_filename

    # store the mapping of IDs
    id_map_filename = os.path.join(kwargs["output"], "tmp", prefix + "id_map.tsv")
    with open(id_map_filename, "w") as out:
        for idx, rep_id in id_map:
            print(idx, rep_id, sep="\t", file=out)
    output_args[prefix + "id_map"] = id_map_filename

    return output_args


def read_folder(folder_path: str, file_extension: str) -> Generator[Tuple[str, str], None, None]:
    """
    Read in all PDB file from a folder and ignore non-PDB files.

    Args:
        folder_path: Path to the folder storing the PDB files
        file_extension: File extension to parse

    Yields:
        Pairs of the PDB files name and the path to the file
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            yield ".".join(filename.split(".")[:-1]), os.path.abspath(os.path.join(folder_path, filename))


def parse_fasta(
        path: str = None,
) -> Dict[str, str]:
    """
    Parse a FASTA file and do some validity checks if requested.

    Args:
        path: Path to the FASTA file

    Returns:
        Dictionary mapping sequences IDs to amino acid sequences
    """
    seq_map = {}

    with open(path, "r") as fasta:
        for line in fasta.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '>':
                entry_id = line[1:].replace(" ", "_")
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map
