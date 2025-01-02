from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromMolFile, MolFromMol2File, MolFromPDBFile, MolFromTPLFile

from datasail.reader.utils import DataSet, read_data, DATA_INPUT, MATRIX_INPUT, read_data_input, read_sdf_file
from datasail.settings import M_TYPE, UNK_LOCATION, FORM_SMILES, LOGGER

if rdkit.__version__ < "2022.09.1":
    from rdkit.Chem import MolFromMol2File
    LOGGER.warning("RDKit version is too old, .xyz, and .mrv files are not supported.")
    MolFromMrvFile, MolFromXYZFile = None, None
elif rdkit.__version__ < "2023.09.1":
    from rdkit.Chem import MolFromMol2File, MolFromXYZFile
    LOGGER.warning("RDKit version is too old, .mrv files are not supported.")
    MolFromMrvFile = None
else:
    from rdkit.Chem import MolFromMol2File, MolFromXYZFile, MolFromMrvFile

mol_reader = {
    "mol": MolFromMolFile,
    "mol2": MolFromMol2File,
    "mrv": MolFromMrvFile,
    "pdb": MolFromPDBFile,
    "tpl": MolFromTPLFile,
    "xyz": MolFromXYZFile,
}


def read_molecule_data(
        data: DATA_INPUT,
        weights: DATA_INPUT = None,
        strats: DATA_INPUT = None,
        sim: MATRIX_INPUT = None,
        dist: MATRIX_INPUT = None,
        inter: Optional[List[Tuple[str, str]]] = None,
        index: Optional[int] = None,
        num_clusters: Optional[int] = None,
        tool_args: str = "",
        detect_duplicates: bool = True,
) -> DataSet:
    """
    Read in molecular data, compute the weights, and distances or similarities of every entity.

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
    dataset = DataSet(type=M_TYPE, format=FORM_SMILES, location=UNK_LOCATION)

    def read_dir(ds: DataSet, path: Path) -> None:
        ds.data = {}
        for file in path.iterdir():
            if file.suffix[1:].lower() != "sdf":
                if (reader := mol_reader[file.suffix[1:].lower()]) is not None:
                    mol = reader(str(file))
                    if mol is not None:
                        if mol.HasProp("_Name"):
                            ds.data[mol.GetProp("_Name")] = Chem.MolToSmiles(mol)
                        else:
                            ds.data[file.stem] = Chem.MolToSmiles(mol)
                else:
                    raise ValueError(f"File type {file.suffix[1:]} is not supported.")
            else:
                ds.data = read_sdf_file(file)

    read_data_input(data, dataset, read_dir)

    dataset = read_data(weights, strats, sim, dist, inter, index, num_clusters, tool_args, dataset)
    if detect_duplicates:
        dataset = remove_molecule_duplicates(dataset)

    return dataset


def remove_molecule_duplicates(dataset: DataSet) -> DataSet:
    """
    Remove duplicates from molecular input data by checking if the input molecules are the same. If a molecule cannot
    be read by RDKit, it will be considered unique and survive the check.

    Args:
        dataset: The dataset to remove duplicates from

    Returns:
        Update arguments as the location of the data might change and an ID-Map file might be added.
    """
    if isinstance(dataset.data[dataset.names[0]], (list, tuple, np.ndarray)):
        # TODO: proper check for duplicate embeddings
        dataset.id_map = {n: n for n in dataset.names}
        return dataset

    # Extract invalid molecules
    non_mols = []
    valid_mols = dict()
    for k, mol in dataset.data.items():
        if mol != mol or mol is None:
            non_mols.append(k)
            continue
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
    """
    Remove duplicate values from the dataset.

    Args:
        dataset: Dataset to remove duplicates from
        data: Data to remove duplicates from

    Returns:
        Updated dataset with removed duplicates
    """
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
