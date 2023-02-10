import cvxpy


def init_variables(num_splits, len_data):
    x = {}
    for s in range(num_splits):
        for i in range(len_data):
            x[i, s] = cvxpy.Variable(boolean=True)
    return x


def init_inter_variables_cluster(num_splits, e_clusters, f_clusters):
    x = {}
    for s in range(num_splits):
        for i in range(len(e_clusters)):
            for j in range(len(f_clusters)):
                x[i, j, s] = cvxpy.Variable(boolean=True)
    return x


def sum_constraint(e_data, x, splits):
    return [sum(x[i, s] for s in range(len(splits))) == 1 for i in range(len(e_data))]


def interaction_constraints(e_data, f_data, inter, x_e, x_f, x_i, s):
    constraints = []
    for i, e1 in enumerate(e_data):
        for j, e2 in enumerate(f_data):
            if (e1, e2) in inter:
                constraints.append(x_i[i, j, s] >= (x_e[i, s] + x_f[j, s] - 1.5))
                constraints.append(x_i[i, j, s] <= (x_e[i, s] + x_f[j, s]) * 0.5)
                constraints.append(x_e[i, s] >= x_i[i, j, s])
                constraints.append(x_f[j, s] >= x_i[i, j, s])
    return constraints


def cluster_sim_dist_constraint(similarities, distances, threshold, num_clusters, x, s):
    constraints = []
    if similarities is not None:
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                constraints.append((x[i, s] - x[j, s]) ** 2 * similarities[i][j] <= threshold)
    else:
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                constraints.append(cvxpy.maximum((x[i, s] + x[j, s]) - 1, 0) * distances[i][j] <= threshold)
    return constraints


def cluster_sim_dist_objective(similarities, distances, num_clusters, x, num_splits):
    if similarities is not None:
        return sum(
            (x[i, s] - x[j, s]) ** 2 * similarities[i][j]
            for i in range(num_clusters) for j in range(i + 1, num_clusters) for s in range(num_splits)
        )
    else:
        return sum(
            cvxpy.maximum((x[i, s] + x[j, s]) - 1, 0) * distances[i][j]
            for i in range(num_clusters) for j in range(i + 1, num_clusters) for s in range(num_splits)
        )
