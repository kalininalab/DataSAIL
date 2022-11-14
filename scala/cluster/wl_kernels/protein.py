import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt

from grakel import Graph

from src.wlk import run_wl_kernel


def list_to_dict(l):
    """Convert list to dict"""
    return {val: i for i, val in enumerate(l)}


node_encoding = {
    "ala": 0, "arg": 1, "asn": 2, "asp": 3, "cys": 4, "gln": 5, "glu": 6, "gly": 7, "his": 8, "ile": 9,
    "leu": 10, "lys": 11, "met": 12, "phe": 13, "pro": 14, "ser": 15, "thr": 16, "trp": 17, "tyr": 18, "val": 19,
}


def dist(p1, p2):
    return sum([(p1[0] - p2[0]) ** 2, (p1[1] - p2[1]) ** 2, (p1[2] - p2[2]) ** 2]) ** (1 / 2)


class Residue:
    """Residue class"""

    def __init__(self, line: str) -> None:
        """

        :param line:
        """
        self.name = line[17:20].strip()
        self.num = int(line[22:26].strip())
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())


class PDBStructure:
    """Structure class"""

    def __init__(self, filename: str) -> None:
        """

        :param filename:
        """
        self.residues = {}
        with open(filename, "r") as in_file:
            for line in in_file.readlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res = Residue(line)
                    self.residues[res.num] = res

    def __len__(self) -> int:
        """

        :return:
        """
        return len(self.residues)

    def __get_coords(self) -> List[Tuple[int, int, int]]:
        """

        :return:
        """
        coords = [(res.num, (res.x, res.y, res.z)) for res in self.residues.values()]
        return coords

    def get_edges(self, threshold: float = 7):
        """
        Get edges of a graph using threshold as a cutoff

        :param threshold:
        :return:
        """
        coords = self.__get_coords()
        return [(coords[i][0], coords[j][0]) for i in range(len(coords)) for j in range(len(coords)) if
                dist(coords[i][1], coords[j][1]) < threshold]

    def get_nodes(self):
        return dict([(res.num, (node_encoding[res.name.lower()])) for i, res in enumerate(self.residues.values())])


def pdb_to_grakel(pdb: Union[str, PDBStructure], threshold: float = 7):
    if isinstance(pdb, str):
        pdb = PDBStructure(pdb)

    tmp_edges = pdb.get_edges(threshold)
    edges = {}
    for start, end in tmp_edges:
        if start not in edges:
            edges[start] = []
        edges[start].append(end)

    return Graph(edges, node_labels=pdb.get_nodes())


def main(pdb_path):
    graphs = []
    for filename in os.listdir(pdb_path):
        graphs.append(pdb_to_grakel(os.path.join(pdb_path, filename)))
    result = run_wl_kernel(graphs)
    plt.pcolormesh(result)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main("./data/pdb_test_data/")
