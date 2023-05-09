import os
from typing import List, Tuple, Dict, Any, Optional

from rdkit.Chem import MolFromSmiles, MolToSmiles

from datasail.reader.utils import read_csv, DataSet, read_data


def read_molecule_data(
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
    Read in molecular data, compute the weights, and distances or similarities of every entity.

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
    dataset = DataSet(type="M")
    if data.lower().endswith(".tsv"):
        dataset.data = dict(read_csv(data))
        dataset.format = "SMILES"
    elif os.path.isdir(data):
        pass
    else:
        raise ValueError()
    dataset.location = data

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def remove_molecule_duplicates(prefix: str, **kwargs) -> Dict[str, Any]:
    """
    Remove duplicates from molecular input data by checking if the input molecules are the same. If a molecule cannot
    be read by RDKit, it will be considered unique and survive the check.

    Args:
       prefix: Prefix of the data. This is either 'e_' or 'f_'
        **kwargs: Arguments for this data input

    Returns:
        Update arguments as teh location of the data might change and an ID-Map file might be added.
    """
    output_args = {prefix + k: v for k, v in kwargs.items()}
    # TODO: turn off rdkit errors and warnings
    if kwargs["data"].lower().endswith(".tsv"):
        input_data = dict(read_csv(kwargs["data"]))
        molecules = {k: MolFromSmiles(v) for k, v in input_data}
    else:
        return output_args

    # Extract invalid molecules
    non_mols = []
    valid_mols = dict()
    for k, mol in molecules:
        if mol is None:
            non_mols.append(k)
        else:
            valid_mols[k] = MolToSmiles(mol)

    id_list = []  # unique ids
    id_map = {}  # mapping of all ids to their representative
    duplicate_found = False
    for idx, seq in valid_mols.items():
        for q_id in id_list:
            if seq == valid_mols[q_id]:
                id_map[idx] = q_id
                duplicate_found = True
        if idx not in id_map:
            id_list.append(idx)
            id_map[idx] = idx

    # no duplicates found, no further action necessary
    if not duplicate_found:
        return output_args

    # update the lists and maps with the "invalid" molecules
    valid_mols.update({k: input_data[k] for k in non_mols})
    id_list += non_mols
    id_map.update({k: k for k in non_mols})

    # store the new SMILES TSV file
    smiles_filename = os.path.abspath(os.path.join(kwargs["output"], "tmp", prefix + "smiles.tsv"))
    with open(smiles_filename, "w") as out:
        for idx in id_list:
            print(idx, valid_mols[idx], sep="\t", file=out)
    output_args[prefix + "data"] = smiles_filename

    # store the mapping of IDs
    id_map_filename = os.path.join(kwargs["output"], "tmp", prefix + "id_map.tsv")
    with open(id_map_filename, "w") as out:
        for idx, rep_id in id_map:
            print(idx, rep_id, sep="\t", file=out)
    output_args[prefix + "id_map"] = id_map_filename

    return output_args
