from typing import List, Tuple

import cvxpy
import numpy as np


def interaction_constraints(e_data, f_data, x_e, x_f, x_i, s):
    constraints = []
    for i in range(len(e_data)):
        for j in range(len(f_data)):
            constraints.append(x_i[s][i, j] >= (x_e[s][:, 0][i] + x_f[s][:, 0][j] - 1.5))
            constraints.append(x_i[s][i, j] <= (x_e[s][:, 0][i] + x_f[s][:, 0][j]) * 0.5)
            constraints.append(x_f[s][:, 0][i] >= x_i[s][i, j])
            constraints.append(x_f[s][:, 0][j] >= x_i[s][i, j])
    return constraints


def cluster_sim_dist_constraint(similarities, distances, threshold, ones, x, s):
    if distances is not None:
        return cvxpy.multiply(
            cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0), distances
        ) <= threshold
    return cvxpy.multiply(((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities) <= threshold


def cluster_sim_dist_objective(similarities, distances, ones, x, splits):
    if distances is not None:
        return cvxpy.sum([cvxpy.sum(cvxpy.multiply(
            cvxpy.maximum((x[s] @ ones) + cvxpy.transpose(x[s] @ ones) - (ones.T @ ones), 0), distances)
        ) for s in range(len(splits))])
    return cvxpy.sum([cvxpy.sum(cvxpy.multiply(
        ((x[s] @ ones) - cvxpy.transpose(x[s] @ ones)) ** 2, similarities
    )) for s in range(len(splits))])
