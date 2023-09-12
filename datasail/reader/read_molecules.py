import os
from typing import List, Tuple, Dict, Any, Optional, Callable, Generator

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromMol2File, MolFromMolFile, MolFromPDBFile, MolFromPNGFile, \
    MolFromTPLFile, MolFromXYZFile

from datasail.cluster.utils import read_molecule_encoding
from datasail.reader.utils import read_csv, DataSet, read_data, DATA_INPUT, MATRIX_INPUT
from datasail.settings import M_TYPE, UNK_LOCATION, FORM_SMILES


mol_reader = {
    "mol2": MolFromMol2File,
    "mol": MolFromMolFile,
    # "sdf": MolFromMol2File,
    "pdb": MolFromPDBFile,
    "png": MolFromPNGFile,
    "tpl": MolFromTPLFile,
    "xyz": MolFromXYZFile,
}


def read_molecule_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        max_sim: float = 1.0,
        max_dist: float = 1.0,
        id_map: Optional[str] = None,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
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
    dataset = DataSet(type=M_TYPE, format=FORM_SMILES, location=UNK_LOCATION)
    if isinstance(data, str):
        if data.lower().endswith(".tsv"):
            dataset.data = dict(read_csv(data))
        elif os.path.isdir(data):
            dataset.data = {}
            for file in os.listdir(data):
                ending = file.split(".")[-1]
                if ending != "sdf":
                    dataset.data[os.path.basename(file)] = mol_reader[ending](os.path.join(data, file))
                else:
                    suppl = Chem.SDMolSupplier(os.path.join(data, file))
                    for i, mol in enumerate(suppl):
                        dataset.data[f"{os.path.basename(file)}_{i}"] = mol
        else:
            raise ValueError()
        dataset.location = data
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError()

    dataset, inter = read_data(weights, sim, dist, max_sim, max_dist, id_map, inter, index, dataset)

    return dataset, inter


def remove_molecule_duplicates(prefix: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Remove duplicates from molecular input data by checking if the input molecules are the same. If a molecule cannot
    be read by RDKit, it will be considered unique and survive the check.

    Args:
        prefix: Prefix of the data. This is either 'e_' or 'f_'
        output_dir: Directory to store data to in case of detected duplicates
        **kwargs: Arguments for this data input

    Returns:
        Update arguments as teh location of the data might change and an ID-Map file might be added.
    """
    output_args = {prefix + k: v for k, v in kwargs.items()}
    if not isinstance(kwargs["data"], str) or not kwargs["data"].lower().endswith(".tsv"):
        return output_args
    # TODO: turn off rdkit errors and warnings
    else:
        input_data = dict(read_csv(kwargs["data"]))
        molecules = {k: read_molecule_encoding(v) for k, v in input_data.items()}

    # Extract invalid molecules
    non_mols = []
    valid_mols = dict()
    for k, mol in molecules.items():
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
    smiles_filename = os.path.abspath(os.path.join(output_dir, "tmp", prefix + "smiles.tsv"))
    pd.DataFrame(
        [(idx, valid_mols[idx]) for idx in id_list], columns=["Representatives", "SMILES"]
    ).to_csv(smiles_filename, sep="\t", columns=["Representatives", "SMILES"], index=False)
    output_args[prefix + "data"] = smiles_filename

    # store the mapping of IDs
    id_map_filename = os.path.join(output_dir, "tmp", prefix + "id_map.tsv")
    pd.DataFrame(
        [(x1, id_map.get(x2, "")) for x1, x2 in id_map.items()], columns=["ID", "Cluster_ID"],
    ).to_csv(id_map_filename, sep="\t", columns=["ID", "Cluster_ID"], index=False)
    output_args[prefix + "id_map"] = id_map_filename

    return output_args
