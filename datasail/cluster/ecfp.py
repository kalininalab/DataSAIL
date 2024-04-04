from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

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

    LOGGER.info("Start ECFP clustering")

    invalid_mols = []
    scaffolds = {}
    for name in dataset.names:
        try:
            mol = Chem.MolFromSmiles(dataset.data[name])
        except:
            mol = None
        # scaffold = read_molecule_encoding(dataset.data[name])
        # if scaffold is None:
        if mol is None:
            LOGGER.warning(f"RDKit cannot parse {name} >{dataset.data[name]}< as a molecule. Skipping.")
            invalid_mols.append(name)
            continue
        scaffolds[name] = mol

    for invalid_name in invalid_mols:
        dataset.names.remove(invalid_name)
        dataset.data.pop(invalid_name)
        poppable = []
        for key, value in dataset.id_map.items():
            if value == invalid_name:
                poppable.append(key)
        for pop in poppable:
            dataset.id_map.pop(pop)
    
    fps = [AllChem.GetMorganFingerprintAsBitVect(scaffolds[name], 2, nBits=1024) for name in dataset.names]
    dataset.cluster_names = dataset.names

    LOGGER.info(f"Reduced {len(dataset.names)} molecules to {len(dataset.cluster_names)}")
    LOGGER.info("Compute Tanimoto Coefficients")

    run(dataset, fps, method)

    dataset.cluster_map = {name: name for name in dataset.names}
