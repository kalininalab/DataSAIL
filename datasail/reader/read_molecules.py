import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Generator, Union, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromMol2File, MolFromMolFile, MolFromPDBFile, MolFromTPLFile, MolFromXYZFile
try:
    from rdkit.Chem import MolFromMrvFile
except ImportError:
    MolFromMrvFile = None

from datasail.reader.utils import read_csv, DataSet, read_data, DATA_INPUT, MATRIX_INPUT
from datasail.settings import M_TYPE, UNK_LOCATION, FORM_SMILES


mol_reader = {
    ".mol": MolFromMolFile,
    ".mol2": MolFromMol2File,
    ".mrv": MolFromMrvFile,
    # "sdf": MolFromMol2File,
    ".pdb": MolFromPDBFile,
    ".tpl": MolFromTPLFile,
    ".xyz": MolFromXYZFile,
}


def read_molecule_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
        tool_args: str = "",
) -> DataSet:
    """
    Read in molecular data, compute the weights, and distances or similarities of every entity.

    Args:
        data: Where to load the data from
        weights: Weight file for the data
        sim: Similarity file or metric
        dist: Distance file or metric
        inter: Interaction, alternative way to compute weights
        index: Index of the entities in the interaction file
        tool_args: Additional arguments for the tool

    Returns:
        A dataset storing all information on that datatype
    """
    dataset = DataSet(type=M_TYPE, format=FORM_SMILES, location=UNK_LOCATION)
    if isinstance(data, Path):
        if data.suffix[1:].lower() == "tsv":
            dataset.data = dict(read_csv(data))
        elif data.is_dir():
            dataset.data = {}
            for file in data.iterdir():
                if file.suffix[1:].lower() != "sdf" and mol_reader[file.suffix[1:].lower()] is not None:
                    dataset.data[file.stem] = mol_reader[file.suffix[1:].lower()](file)
                else:
                    suppl = Chem.SDMolSupplier(file)
                    for i, mol in enumerate(suppl):
                        dataset.data[f"{file.stem}_{i}"] = mol
        else:
            raise ValueError()
        dataset.location = data
    elif isinstance(data, Union[list, tuple]) and isinstance(data[0], Iterable) and len(data[0]) == 2:
        dataset.data = dict(data)
    elif isinstance(data, dict):
        dataset.data = data
    elif isinstance(data, Callable):
        dataset.data = data()
    elif isinstance(data, Generator):
        dataset.data = dict(data)
    else:
        raise ValueError()

    dataset = read_data(weights, sim, dist, inter, index, tool_args, dataset)
    dataset = remove_molecule_duplicates(dataset)

    return dataset


def remove_molecule_duplicates(dataset: DataSet) -> DataSet:
    """
    Remove duplicates from molecular input data by checking if the input molecules are the same. If a molecule cannot
    be read by RDKit, it will be considered unique and survive the check.

    Args:
        dataset: The dataset to remove duplicates from

    Returns:
        Update arguments as teh location of the data might change and an ID-Map file might be added.
    """

    # Extract invalid molecules
    non_mols = []
    valid_mols = dict()
    for k, mol in dataset.data.items():
        molecule = Chem.MolFromSmiles(mol)
        if molecule is None:
            non_mols.append(k)
        else:
            valid_mols[k] = Chem.MolToInchi(molecule)

    return remove_duplicate_values(dataset, valid_mols)

    # update the lists and maps with the "invalid" molecules
    # valid_mols.update({k: input_data[k] for k in non_mols})
    # id_list += non_mols
    # id_map.update({k: k for k in non_mols})

    # return id_map


def remove_duplicate_values(dataset, data) -> DataSet:
    tmp = dict()
    for idx, mol in data.items():
        if mol not in tmp:
            tmp[mol] = []
        tmp[mol].append(idx)
    dataset.id_map = {idx: ids[0] for ids in tmp.values() for idx in ids}

    ids = set()
    for i, name in enumerate(dataset.names):
        if name not in dataset.id_map:
            ids.add(i)
            continue
        if dataset.id_map[name] != name:
            dataset.weights[dataset.id_map[name]] += dataset.weights[name]
            del dataset.data[name]
            del dataset.weights[name]
            ids.add(i)
    dataset.names = [name for i, name in enumerate(dataset.names) if i not in ids]
    ids = list(ids)
    if isinstance(dataset.similarity, np.ndarray):
        dataset.similarity = np.delete(dataset.similarity, ids, axis=0)
        dataset.similarity = np.delete(dataset.similarity, ids, axis=1)
    if isinstance(dataset.distance, np.ndarray):
        dataset.distance = np.delete(dataset.distance, ids, axis=0)
        dataset.distance = np.delete(dataset.distance, ids, axis=1)

    return dataset
