import logging
import pickle
from typing import Tuple, List, Dict

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric

from datasail.reader.utils import DataSet


def run_ecfp(dataset: DataSet) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Compute 1024Bit-ECPFs for every molecule in the dataset and then compute pairwise Tanimoto-Scores of them.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    if dataset.type != "M":
        raise ValueError("ECFP with Tanimoto-scores can only be applied to molecular data.")

    scaffolds = {}
    logging.info("Start ECFP clustering")

    invalid_mols = []
    for name in dataset.names:
        scaffold = Chem.MolFromSmiles(dataset.data[name])
        if scaffold is None:
            logging.warning(f"RDKit cannot parse {name} ({dataset.data[name]})")
            invalid_mols.append(name)
            continue
        scaffolds[name] = MakeScaffoldGeneric(scaffold)
    for invalid_name in invalid_mols:
        dataset.names.remove(invalid_name)
        dataset.data.pop(invalid_name)

    fps = []
    cluster_names = list(set(Chem.MolToSmiles(s) for s in list(scaffolds.values())))
    for scaffold in cluster_names:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(scaffold), 2, nBits=1024))

    logging.info(f"Reduced {len(dataset.names)} molecules to {len(cluster_names)}")

    logging.info("Compute Tanimoto Coefficients")

    count = len(cluster_names)
    sim_matrix = np.zeros((count, count))
    for i in range(count):
        if i % 100 == 0:
            print(f"\r{i + 1} / {count}", end="")
        sim_matrix[i, i] = 1
        sim_matrix[i, :i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        sim_matrix[:i, i] = sim_matrix[i, :i]

    cluster_map = dict((name, Chem.MolToSmiles(scaffolds[name])) for name in dataset.names)
    fig, ax = plt.subplots()

    heatmap(sim_matrix, ax=ax, cmap="YlGn")
    fig.tight_layout()
    plt.savefig("heatmap_mibig.png")
    plt.clf()

    return cluster_names, cluster_map, sim_matrix


def heatmap(data, ax=None, cbar_kw=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    return im, cbar
