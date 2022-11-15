import logging
from typing import Dict, Tuple, Set, List, Optional

from ortools.sat.python import cp_model
from sortedcontainers import SortedList


def solve_mkp_ilp_ic(
        drugs: SortedList,
        drug_weights: Dict[str, float],
        proteins: SortedList,
        protein_weights: Dict[str, float],
        inter: Set[Tuple[str, str]],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    # Create the mip solver with the algorithm backend.
    model = cp_model.CpModel()
    if model is None:
        print(f"SAT solver unavailable.")
        return None

    # Variables.
    # x[i, b] = 1 if item i is packed in bin b.
    x_d = {}
    for i in range(len(drugs)):
        for b in range(len(splits)):
            x_d[i, b] = model.NewBoolVar(f'x_d_{i}_{b}')
    x_p = {}
    for j in range(len(proteins)):
        for b in range(len(splits)):
            x_p[j, b] = model.NewBoolVar(f'x_p_{j}_{b}')
    x_e = {}
    x_dp = {}
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                x_e[i, j] = model.NewBoolVar(f'x_e_{i}_{j}')
                for b in range(len(splits)):
                    x_dp[i, j, b] = model.NewBoolVar(f"x_dp_{i}_{j}_{b}")

    for i in range(len(drugs)):
        model.Add(sum(x_d[i, b] for b in range(len(splits))) <= 1)
    for j in range(len(proteins)):
        model.Add(sum(x_p[j, b] for b in range(len(splits))) <= 1)

    for b in range(len(splits)):
        model.Add(
            # int(splits[b] * len(inter) * (1 - limit)) <=
            sum(x_d[i, b] * drug_weights[drugs[i]] for i in range(len(drugs))) <=
            int(splits[b] * len(inter) * (1 + limit))
        )
        model.Add(
            # int(splits[b] * len(inter) * (1 + limit)) <=
            sum(x_p[j, b] * protein_weights[proteins[j]] for j in range(len(proteins))) <=
            int(splits[b] * len(inter) * (1 + limit))
        )

    for b in range(len(splits)):
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    model.Add(x_dp[i, j, b] == 1).OnlyEnforceIf(x_d[i, b]).OnlyEnforceIf(x_p[j, b])
                    model.Add(x_d[i, b] == 1).OnlyEnforceIf(x_dp[i, j, b])
                    model.Add(x_p[j, b] == 1).OnlyEnforceIf(x_dp[i, j, b])

    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                model.Add(x_e[i, j] == sum(x_dp[i, j, b] for b in range(len(splits))))

    model.Maximize(
        sum(x_d[i, b] for i in range(len(drugs)) for b in range(len(splits))) +
        sum(x_p[j, b] for j in range(len(proteins)) for b in range(len(splits))) +
        sum(x_e[i, j] for i in range(len(drugs)) for j in range(len(proteins)) if (i, j) in inter)
    )

    solver = cp_model.CpSolver()
    if max_sec != -1:
        solver.parameters.max_time_in_seconds = max_sec
    status = solver.Solve(model)

    output = [[], {}, {}]
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i, drug in enumerate(drugs):
            for b in range(len(splits)):
                if solver.Value(x_d[i, b]) > 0:
                    output[1][drug] = names[b]
            if drug not in output[1]:
                output[1][drug] = "not selected"
        for j, protein in enumerate(proteins):
            for b in range(len(splits)):
                if solver.Value(x_p[j, b]) > 0:
                    output[2][protein] = names[b]
            if protein not in output[2]:
                output[2][protein] = "not selected"
        count = 0
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    count += 1
                    print(drug, protein)
                    if solver.Value(x_e[i, j]) > 0:
                        output[0].append((drug, protein, output[1][drug]))
                    else:
                        output[0].append((drug, protein, "not selected"))
        print(count)
        return output
    else:
        logging.warning(
            'The ILP cannot be solved. Please consider a relaxed clustering, i.e., more clusters, or a higher limit.'
        )
    return None
