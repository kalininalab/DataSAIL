from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit.Chem.rdchem import MolSanitizeException

from datasail.cluster.utils import read_molecule_encoding
from datasail.cluster.vectors import run, SIM_OPTIONS
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER


def run_ecfp(dataset: DataSet, method: SIM_OPTIONS = "tanimoto") -> None:
    """
    Compute 1024Bit-ECPFs for every molecule in the dataset and then compute pairwise Tanimoto-Scores of them.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for
        method: The similarity measure to use. Default is "Tanimoto".
    """
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    if dataset.type != "M":
        raise ValueError("ECFP with Tanimoto-scores can only be applied to molecular data.")

    scaffolds = {}
    LOGGER.info("Start ECFP clustering")

    invalid_mols = []
    for name in dataset.names:
        scaffold = read_molecule_encoding(dataset.data[name])
        if scaffold is None:
            bo, bc = "{", "}"
            LOGGER.warning(f"RDKit cannot parse {name} {bo}{dataset.data[name]}{bc}")
            invalid_mols.append(name)
            continue
        try:
            scaffolds[name] = MakeScaffoldGeneric(scaffold)
        except MolSanitizeException:
            LOGGER.warning(f"RDKit cannot parse {name} ({dataset.data[name]})")
            invalid_mols.append(name)
            continue
    for invalid_name in invalid_mols:  # obsolete code?
        dataset.names.remove(invalid_name)
        dataset.data.pop(invalid_name)
        poppable = []
        for key, value in dataset.id_map.items():
            if value == invalid_name:
                poppable.append(key)
        for pop in poppable:
            dataset.id_map.pop(pop)

    fps = []
    dataset.cluster_names = list(set(Chem.MolToSmiles(s) for s in list(scaffolds.values())))
    for scaffold in dataset.cluster_names:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(scaffold), 2, nBits=1024))

    LOGGER.info(f"Reduced {len(dataset.names)} molecules to {len(dataset.cluster_names)}")
    LOGGER.info("Compute Tanimoto Coefficients")

    run(dataset, fps, method)

    dataset.cluster_map = dict((name, Chem.MolToSmiles(scaffolds[name])) for name in dataset.names)
