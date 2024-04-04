from pathlib import Path
from typing import Dict, Tuple, List, Union
import math

from grakel import Graph, WeisfeilerLehman, VertexHistogram
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

from datasail.reader.utils import DataSet
from datasail.settings import LOGGER, MAX_PATH

Point = Tuple[float, float, float]

node_encoding = {
    "ala": 0, "arg": 1, "asn": 2, "asp": 3, "cys": 4, "gln": 5, "glu": 6, "gly": 7, "his": 8, "ile": 9,
    "leu": 10, "lys": 11, "met": 12, "phe": 13, "pro": 14, "ser": 15, "thr": 16, "trp": 17, "tyr": 18, "val": 19,
}


def run_wlk(dataset: DataSet, n_iter: int = 4) -> None:
    """
    Run Weisfeiler-Lehman kernel-based cluster on the input. As a result, every molecule will form its own cluster

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for
        n_iter: number of iterations in Weisfeiler-Lehman kernels
    """
    if dataset.type != "M":
        raise ValueError("ECFP with Tanimoto-scores can only be applied to molecular data.")

    LOGGER.info("Start WLK clustering")

    sample = list(dataset.data.values())[0]
    if isinstance(sample, Path) or (len(sample) < MAX_PATH and Path(sample).is_file()):  # read PDB files into grakel graph objects
        graphs = [pdb_to_grakel(Path(dataset.data[name])) for name in dataset.names]
    else:  # read molecules from SMILES to grakel graph objects
        graphs = [mol_to_grakel(MolFromSmiles(dataset.data[name])) for name in dataset.names]

    # compute similarity metric and the mapping from element names to cluster names
    dataset.cluster_names = dataset.names
    dataset.cluster_similarity = run_wl_kernel(graphs, n_iter)
    dataset.cluster_map = dict((name, name) for name in dataset.names)


def run_wl_kernel(graph_list: List[Graph], n_iter: int = 4) -> np.ndarray:
    """
    Run the Weisfeiler-Lehman algorithm on the list of input graphs.

    Args:
        graph_list: List of grakel-graphs to run pairwise similarity search on
        n_iter: number of iterations in Weisfeiler-Lehman kernels

    Returns:
        Symmetric 2D-numpy array storing pairwise similarities of the input graphs
    """
    gk = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
    gk.fit_transform(graph_list)
    result = gk.transform(graph_list)

    return result


def mol_to_grakel(mol: Chem.Mol) -> Graph:
    """
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
    return Graph(edges, node_labels=nodes)


class PDBStructure:
    def __init__(self, filename: Path) -> None:
        """
        Read the $C_{\alpha}$ atoms from a PDB file.

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
        return dict(
            [(res.num, (node_encoding.get(res.name.lower(), 20))) for i, res in enumerate(self.residues.values())])


def pdb_to_grakel(pdb: Union[Path, PDBStructure], threshold: float = 7) -> Graph:
    """
    Convert a PDB file into a grakel graph to compute WLKs over them.

    Args:
        pdb: Either PDB structure or filepath to PDB file
        threshold: Distance threshold to apply when computing the graphs

    Returns:
        A grakel graph based on the PDB structure
    """
    if isinstance(pdb, Path):
        pdb = PDBStructure(pdb)

    tmp_edges = pdb.get_edges(threshold)
    edges = {}
    for start, end in tmp_edges:
        if start not in edges:
            edges[start] = []
        edges[start].append(end)

    return Graph(edges, node_labels=pdb.get_nodes())


class Residue:
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
