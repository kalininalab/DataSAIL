import numpy as np
from grakel import WeisfeilerLehman, VertexHistogram, Graph
from typing import Dict, List


def to_grakel(edges: Dict[int, List[int]], node_labels: Dict[int, List]):
    return Graph(edges, node_labels=node_labels)


def run_wl_kernel(graph_list: List[Graph]) -> np.ndarray:
    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    gk.fit_transform(graph_list)
    result = gk.transform(graph_list)

    return result
