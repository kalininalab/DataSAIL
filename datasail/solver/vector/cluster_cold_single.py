from typing import List, Union, Optional, Dict

import cvxpy
import numpy as np

from datasail.solver.utils import solve
from datasail.solver.vector.utils import cluster_sim_dist_constraint, cluster_sim_dist_objective


def solve_ccs_bqp(
        e_clusters: List[Union[str, int]],
        e_weights: List[float],
        e_similarities: Optional[np.ndarray],
        e_distances: Optional[np.ndarray],
        e_threshold: float,
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:

    ones = np.ones((1, len(e_clusters)))

    x_e = [cvxpy.Variable((len(e_clusters), 1), boolean=True) for _ in range(len(splits))]

    e_t = np.full((len(e_clusters), len(e_clusters)), e_threshold)
    min_lim = [int(split * sum(e_weights) * (1 - limit)) for split in splits]
    max_lim = [int(split * sum(e_weights) * (1 + limit)) for split in splits]

    constraints = [
        cvxpy.sum([a[:, 0] for a in x_e]) == np.ones((len(e_clusters))),
    ]
    for s, split in enumerate(splits):
        constraints += [
            min_lim[s] <= cvxpy.sum(cvxpy.multiply(x_e[s][:, 0], e_weights)),
            cvxpy.sum(cvxpy.multiply(x_e[s][:, 0], e_weights)) <= max_lim[s],
            cluster_sim_dist_constraint(e_similarities, e_distances, e_t, ones, x_e, s)
        ]

    size_loss = sum(
        (cvxpy.sum(cvxpy.multiply(e_weights, x_e[s][:, 0])) - split * sum(e_weights)) ** 2
        for s, split in enumerate(splits)
    )

    e_loss = cluster_sim_dist_objective(e_similarities, e_distances, ones, x_e, splits)

    alpha = 0.5
    solve(alpha * size_loss + e_loss, constraints, max_sec, len(x_e))

    return dict(
        (e, names[s]) for s in range(len(splits)) for i, e in enumerate(e_clusters) if x_e[s][i, 0].value > 0.1
    )
