import os
import pickle
from typing import Dict, Tuple, List, Union
import math

import grakel
import numpy as np
from grakel import Graph, WeisfeilerLehman, VertexHistogram
from matplotlib import pyplot as plt
from rdkit.Chem import MolFromSmiles

from datasail.cluster.foldseek import run_foldseek
from datasail.reader.utils import DataSet

Point = Tuple[float, float, float]

node_encoding = {
    "ala": 0, "arg": 1, "asn": 2, "asp": 3, "cys": 4, "gln": 5, "glu": 6, "gly": 7, "his": 8, "ile": 9,
    "leu": 10, "lys": 11, "met": 12, "phe": 13, "pro": 14, "ser": 15, "thr": 16, "trp": 17, "tyr": 18, "val": 19,
}


def run_wlk(dataset: DataSet) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Run Weisfeiler-Lehman kernel-based cluster on the input. As a result, every molecule will form its own cluster

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for

    Returns:
        A tuple containing
          - the names of the clusters (cluster representatives)
          - the mapping from cluster members to the cluster names (cluster representatives)
          - the similarity matrix of the clusters (a symmetric matrix filled with 1s)
    """
    if os.path.isfile(list(dataset.data.values())[1]):  # read PDB files into grakel graph objects
        cluster_names, graphs = list(zip(*((name, pdb_to_grakel(pdb_path)) for name, pdb_path in dataset.data.items())))
    else:  # read molecules from SMILES to grakel graph objects
        cluster_names, graphs = list(
            zip(*((name, mol_to_grakel(MolFromSmiles(mol))) for name, mol in dataset.data.items())))

    # compute similarity metric and the mapping from element names to cluster names
    cluster_sim = run_wl_kernel(graphs)
    cluster_map = dict((name, name) for name, _ in dataset.data.items())

    return cluster_names, cluster_map, cluster_sim


def run_wl_kernel(graph_list: List[Graph]) -> np.ndarray:
    """
    Run the Weisfeiler-Lehman algorithm on the list of input graphs.

    Args:
        graph_list: List of grakel-graphs to run pairwise similarity search on

    Returns:
        Symmetric 2D-numpy array storing pairwise similarities of the input graphs
    """
    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    gk.fit_transform(graph_list)
    result = gk.transform(graph_list)

    return result


def mol_to_grakel(mol) -> grakel.graph.Graph:
    r"""
    Convert an RDKit molecule into a grakel graph to apply Weisfeiler-Lehman kernels later.

    Args:
        mol: RDKit Molecule

    Returns:
        grakel graph object
    """
    # grakel requires a dict of adjacency lists with each node as a key and for every node a node feature (atom type)
    nodes = {}
    edges = {}

    # for every node, insert the atom type into a dict and initialize the adjacency matrices for each node
    for atom in mol.GetAtoms():
        nodes[atom.GetIdx()] = atom.GetAtomicNum()
        edges[atom.GetIdx()] = []

    # for every bond in the molecule insert the nodes into the corresponding adjacency lists
    for edge in mol.GetBonds():
        edges[edge.GetBeginAtomIdx()].append(edge.GetEndAtomIdx())
        edges[edge.GetEndAtomIdx()].append(edge.GetBeginAtomIdx())

    # create the final grakel graph from it
    return grakel.Graph(edges, node_labels=nodes)


class PDBStructure:
    """Structure class"""

    def __init__(self, filename: str) -> None:
        """
        Read the $C_\alpha$ atoms from a PDB file.

        Args:
            filename: PDB filename to read from
        """
        self.residues = {}
        with open(filename, "r") as in_file:
            for line in in_file.readlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res = Residue(line)
                    self.residues[res.num] = res

    def get_edges(self, threshold: float = 7) -> List[Tuple[int, int]]:
        """
        Get edges for the graph representation of this PDB structure based on the distance of the C-alpha atoms

        Args:
            threshold: Distance threshold to accept an edge

        Returns:
            A list of edges given by their residue number
        """
        coords = [(res.num, (res.x, res.y, res.z)) for res in self.residues.values()]
        return [(coords[i][0], coords[j][0]) for i in range(len(coords)) for j in range(len(coords)) if
                math.dist(coords[i][1], coords[j][1]) < threshold]

    def get_nodes(self) -> Dict[int, int]:
        """
        Get the nodes as a map from their residue id to a numerical encoding of the represented amino acid.

        Returns:
            Dict mapping residue ids to a numerical encodings of the represented amino acids
        """
        return dict([(res.num, (node_encoding.get(res.name.lower(), 20))) for i, res in enumerate(self.residues.values())])


def pdb_to_grakel(pdb: Union[str, PDBStructure], threshold: float = 7) -> grakel.graph.Graph:
    """
    Convert a PDB file into a grakel graph to compute WLKs over them.

    Args:
        pdb: Either PDB structure or filepath to PDB file
        threshold: Distance threshold to apply when computing the graphs

    Returns:
        A grakel graph based on the PDB structure
    """
    pdb_str = pdb
    if isinstance(pdb, str):
        pdb = PDBStructure(pdb)

    tmp_edges = pdb.get_edges(threshold)
    edges = {}
    for start, end in tmp_edges:
        if start not in edges:
            edges[start] = []
        edges[start].append(end)

    if len(edges) < 10 or len(pdb.get_nodes()) < 10:
        print(len(edges), "|", len(pdb.get_nodes()))
        print("\t", pdb_str)

    return Graph(edges, node_labels=pdb.get_nodes())


class Residue:
    """Residue class"""

    def __init__(self, line: str) -> None:
        """
        Read in the important information for a residue based on the line of the C-alpha atom.

        Args:
            line: Line to read from.
        """
        self.name = line[17:20].strip()
        self.num = int(line[22:26].strip())
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())


if __name__ == '__main__':
    # path = "/scratch/SCRATCH_SAS/Olga/TMalign/PDBs/"
    # path = "../../tests/data/pipeline/pdbs/"
    path = "/scratch/SCRATCH_SAS/Olga/TMalign/SCOPe_40/"

    # wlk_matrix = run_wl_kernel([pdb_to_grakel(path + name) for name in os.listdir(path)])
    # pickle.dump(wlk_matrix, open("wlk_matrix.pkl", "wb"))
    # np.savetxt('wlk_matrix.tsv', wlk_matrix, delimiter="\t")
    wlk_matrix = pickle.load(open("wlk_matrix.pkl", "rb"))

    _, _, fs_matrix = run_foldseek(DataSet(names=os.listdir(path), location=path))
    pickle.dump(wlk_matrix, open("fs_matrix.pkl", "wb"))
    np.savetxt('fs_matrix.tsv', fs_matrix, delimiter="\t")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.matshow(wlk_matrix)
    ax1.set_xlabel("WL Kernel")
    ax2.matshow(fs_matrix)
    ax2.set_xlabel("FoldSeek")
    plt.savefig("matrices.png")

    print(np.corrcoef(wlk_matrix.flatten(), fs_matrix.flatten()))
