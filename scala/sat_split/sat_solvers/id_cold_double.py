import logging
from typing import Dict, Tuple, Set, List, Optional

from ortools.sat.python import cp_model

from scala.sat_split.sat_solvers.utils import STATUS, SolutionTracker


def solve_ic_sat(
        drugs: list,
        proteins: list,
        inter: Set[Tuple[str, str]],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Tuple[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    """
    Split a dataset of proteins and drugs in a cold-split manner that makes sure, no drug or protein appears in two
    splits.

    Args:
        drugs: list of drug names
        proteins: list of protein names
        inter: list of pairs of one drug and one protein forming the interactions of this dataset
        limit: percentage by how much the limits of the split sizes might be exceeded
        splits: list of sizes of the split in percent of the total number of interactions
        names: names of the splits in the order of their appearance in the splits list
        max_sec: maximal number of second to spend on solving the problem
        max_sol: maximal number of solutions to consider when solving the problem

    Returns:
        If the problem could not be solved, None. Otherwise, a tuple of
            * a list of tuples of drugs, proteins, and split names assigning the interactions
            * a dictionary assigning the drug names to their split
            * a dictionary assigning the protein names to their split
    """
    model = cp_model.CpModel()
    if model is None:
        logging.error(f"SAT solver not available.")
        return None

    # variables
    # One boolean variable per [drug|protein|interaction] and split pair
    x_d = {}
    for b in range(len(splits)):
        for i in range(len(drugs)):
            x_d[i, b] = model.NewBoolVar(f'x_d_{i}_{b}')
    x_p = {}
    for b in range(len(splits)):
        for j in range(len(proteins)):
            x_p[j, b] = model.NewBoolVar(f'x_p_{j}_{b}')
    x_e = {}
    for b in range(len(splits)):
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    x_e[i, j, b] = model.NewBoolVar(f'x_e_{i}_{j}_{b}')

    # Constraints
    # Assure that every drug or protein is assigned to exactly one cluster and the edges to at most one
    for i in range(len(drugs)):
        model.Add(sum(x_d[i, b] for b in range(len(splits))) == 1)
    for j in range(len(proteins)):
        model.Add(sum(x_p[j, b] for b in range(len(splits))) == 1)
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                model.Add(sum(x_e[i, j, b] for b in range(len(splits))) <= 1)

    for b in range(len(splits)):
        # Assure the number of edges in the splits does not exceed the limits
        var = sum(
            x_e[i, j, b] for i in range(len(drugs)) for j in range(len(proteins)) if (drugs[i], proteins[j]) in inter
        )
        model.Add(int(splits[b] * len(inter) * (1 - limit)) < var)
        model.Add(var < int(splits[b] * len(inter) * (1 + limit)))

        # Assure that for all drugs i, proteins j, splits b: (i, j) in b <=> i in b and j in b holds
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    model.Add(x_e[i, j, b] == 1).OnlyEnforceIf(x_d[i, b]).OnlyEnforceIf(x_p[j, b])

                    model.Add(x_d[i, b] == 1).OnlyEnforceIf(x_e[i, j, b])
                    model.Add(x_p[j, b] == 1).OnlyEnforceIf(x_e[i, j, b])

    # Maximize the number of edges in the final dataset
    model.Maximize(
        sum(
            x_e[i, j, b]
            for i in range(len(drugs))
            for j in range(len(proteins))
            for b in range(len(splits))
            if (drugs[i], proteins[j]) in inter
        )
    )

    logging.info("Start optimizing")

    # set up the solver and set constraints for time and number of considered solutions on the solver
    solver = cp_model.CpSolver()
    if max_sec != -1:
        solver.parameters.max_time_in_seconds = max_sec
    if max_sol != -1:
        status = solver.Solve(model, solution_callback=SolutionTracker(max_sol))
    else:
        status = solver.Solve(model)

    logging.info(f"Problem status: {STATUS[status]}")

    # report the found solution
    output = ([], {}, {})
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i, drug in enumerate(drugs):
            for b in range(len(splits)):
                if solver.Value(x_d[i, b]) > 0:
                    output[1][drugs[i]] = names[b]
        for j, protein in enumerate(proteins):
            for b in range(len(splits)):
                if solver.Value(x_p[j, b]) > 0:
                    output[2][proteins[j]] = names[b]
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    for b in range(len(splits)):
                        if solver.Value(x_e[i, j, b]) > 0:
                            output[0].append((drugs[i], proteins[j], names[b]))
                    if sum(solver.Value(x_e[i, j, b]) for b in range(len(splits))) == 0:
                        output[0].append((drugs[i], proteins[j], "not selected"))
        return output
    else:
        logging.warning(
            'The SAT problem cannot be solved. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
    return None
