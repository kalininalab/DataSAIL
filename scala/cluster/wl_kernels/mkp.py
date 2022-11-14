from random import shuffle

import grakel
from rdkit import Chem
import numpy as np
import networkx as nx
from ortools.linear_solver import pywraplp

from scala.cluster.wl_kernels.wlk import run_wl_kernel


def mol_to_grakel(mol):
    r"""
    Convert an RDKit molecule into a grakel graph to apply Weisfeiler-Lehman kernels later.

    Args:
        mol: rdkit Molecule

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


def node_disjoint_cliques(g):
    r"""
    Generate all maximal, node-disjoint cliques from a graph

    Args:
        g: networkx graph

    Returns:
        A generator of all the cliques sorted from biggest to smallest
    """
    while len(g.nodes) > 0:
        # find the currently maximal clique, return it and delete the corresponding nodes
        clique = nx.max_weight_clique(g, weight=None)[0]
        yield clique
        g.remove_nodes_from(clique)


def solve_mkp(clusters):
    r"""
    Wrapper method implementing the MKP into an integer linear problem (ILP) to be solved approximately.

    Args:
        clusters: List of clusters, i.e. list of lists integers referring to the nodes in the similarity graph

    Returns:
        List of splits with node ids of similarity graph
    """
    shuffle(clusters)
    data = {
        "weights": [len(cluster) for cluster in clusters],
        "values": [len(cluster) for cluster in clusters],
        "num_items": len(clusters),
        "all_items": range(len(clusters)),
        "bin_capacities": [
            0.7 * sum([len(c) for c in clusters]) * 1.05,  # bin containing train split clusters
            0.2 * sum([len(c) for c in clusters]) * 1.05,  # bin containing validation split clusters
            0.1 * sum([len(c) for c in clusters]) * 1.05   # bin containing test split clusters
        ],
        "num_bins": 3,
        "all_bins": range(3),
    }

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        print('SCIP solver unavailable.')
        return

    # Variables.
    # x[i, b] = 1 if item i is packed in bin b.
    x = {}
    for i in data['all_items']:
        for b in data['all_bins']:
            x[i, b] = solver.BoolVar(f'x_{i}_{b}')

    # Constraints.
    # Each item is assigned to at most one bin.
    for i in data['all_items']:
        solver.Add(sum(x[i, b] for b in data['all_bins']) <= 1)

    # The amount packed in each bin cannot exceed its capacity.
    for b in data['all_bins']:
        solver.Add(
            sum(x[i, b] * data['weights'][i]
                for i in data['all_items']) <= data['bin_capacities'][b])

    # Objective.
    # Maximize total value of packed items.
    objective = solver.Objective()
    for i in data['all_items']:
        for b in data['all_bins']:
            objective.SetCoefficient(x[i, b], data['values'][i])
    objective.SetMaximization()

    status = solver.Solve()

    bins = [[], [], []]
    if status == pywraplp.Solver.OPTIMAL:
        for b in data['all_bins']:
            for i in data['all_items']:
                if x[i, b].solution_value() > 0:
                    bins[b] += clusters[i]
        return bins
    else:
        print('The problem does not have an optimal solution.')
    return None, None, None


def main():
    # read the glycans into a list of grakel graphs
    valid = []
    with open("data/lig.tsv", "r") as data:
        graphs = []
        for i, line in enumerate(data.readlines()):
            parts = line.strip().split("\t")
            smiles = parts[1]
            # here one can think of using scaffolds instead of the actual molecules
            # maybe scaffolds invalidate the idea of WL kernels?
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and len(smiles) > 10:
                valid.append(parts[0])
                graphs.append(mol_to_grakel(mol))

    # compute a matrix of pairwise graph similarities using Weisfeiler-Lehman kernels
    results = run_wl_kernel(graphs)
    avg = np.average(results)

    # generate a fully connected graph with nodes being molecules and edges their pairwise similarity
    g = nx.Graph()
    for x in range(len(results)):
        for y in range(len(results[x])):
            g.add_edge(x, y, val=results[x, y])

    # filter out all edges that are considered to be not similar to be remove from the graph
    low_edges = list(filter(lambda e: e[2] > avg * 0.75, [e for e in g.edges.data('val')]))
    g.remove_edges_from(low_edges)

    # find the maximal, node-disjoint vertices in the graph and use them as clusters
    i = 0
    cliques = []
    for clique in node_disjoint_cliques(g):
        print(i, "\t", len(clique))
        i += 1
        cliques.append(clique)

    # once having the clusters, interpret this as a multiple knapsacks problem to be solved
    train_ids, val_ids, test_ids = solve_mkp(cliques)
    assert train_ids is not None and val_ids is not None and test_ids is not None

    return [valid[i] for i in train_ids], [valid[i] for i in val_ids], [valid[i] for i in test_ids]


if __name__ == '__main__':
    train_ids, val_ids, test_ids = main()
    print("Train-Set:\n" + "\n".join(train_ids))
    print("Valid-Set:\n" + "\n".join(val_ids))
    print("Test-Set:\n" + "\n".join(test_ids))
