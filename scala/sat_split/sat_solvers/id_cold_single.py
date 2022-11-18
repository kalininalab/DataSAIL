import logging
from typing import List, Dict, Optional

from ortools.sat.python import cp_model
from sortedcontainers import SortedList

from scala.sat_split.sat_solvers.utils import STATUS, SolutionTracker


def solve_icx_sat(
        molecules: SortedList,
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    model = cp_model.CpModel()
    if model is None:
        print(f"SAT solver unavailable.")
        return None

    # Variables
    x = {}
    for i in range(len(molecules)):
        for b in range(len(splits)):
            x[i, b] = model.NewBoolVar(f'x_{i}_{b}')

    # Constraints.
    # Each item is assigned to at most one bin.
    for i in range(len(molecules)):
        model.Add(sum(x[i, b] for b in range(len(splits))) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for b in range(len(splits)):
        var = sum(x[i, b] * weights[i] for i in range(len(molecules)))
        model.Add(int(splits[b] * sum(weights) * (1 - limit)) <= var)
        model.Add(var <= int(splits[b] * sum(weights) * (1 + limit)))

    model.Maximize(
        1
    )

    logging.info("Start optimizing")
    solver = cp_model.CpSolver()
    if max_sec != -1:
        solver.parameters.max_time_in_seconds = max_sec
    if max_sol != -1:
        status = solver.Solve(model, solution_callback=SolutionTracker(max_sol))
    else:
        status = solver.Solve(model)

    output = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logging.info(STATUS[status])
        for i in range(len(molecules)):
            for b in range(len(splits)):
                if solver.Value(x[i, b]) > 0:
                    output[molecules[i]] = names[b]
        return output
    else:
        logging.warning(
            'The SAT problem cannot be solved. Please consider relaxing split restrictions, '
            'e.g., less splits, or a higher tolerance level for exceeding cluster limits.'
        )
    return None
