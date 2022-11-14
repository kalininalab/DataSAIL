import logging
from typing import List, Dict, Optional

from ortools.linear_solver import pywraplp
from sortedcontainers import SortedList


ALGORITHM = "GLOP"


def solve_mpk_ilp_icx(
        molecules: SortedList,
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    # np.random.shuffle(molecules)
    d = {
        "weights": weights,
        "values": weights,
        "num_items": len(molecules),
        "all_items": range(len(molecules)),
        "bin_capacities": [s * sum(weights) * (1 + limit) for s in splits],
        "num_bins": len(splits),
        "all_bins": range(len(splits)),
    }

    # Create the mip solver with the algorithm backend.
    solver = pywraplp.Solver.CreateSolver(ALGORITHM)
    if solver is None:
        print(f"{ALGORITHM} solver unavailable.")
        return None

    # Variables.
    # x[i, b] = 1 if item i is packed in bin b.
    x = {}
    for i in d['all_items']:
        for b in d['all_bins']:
            x[i, b] = solver.BoolVar(f'x_{i}_{b}')

    # Constraints.
    # Each item is assigned to at most one bin.
    for i in d['all_items']:
        solver.Add(sum(x[i, b] for b in d['all_bins']) <= 1)

    # The amount packed in each bin cannot exceed its capacity.
    for b in d['all_bins']:
        solver.Add(sum(x[i, b] * d['weights'][i] for i in d['all_items']) <= d['bin_capacities'][b])

    # Objective. Maximize total value of packed items.
    objective = solver.Objective()
    for i in d['all_items']:
        for b in d['all_bins']:
            objective.SetCoefficient(x[i, b], d['values'][i])
    objective.SetMaximization()

    solver.set_time_limit(max_sec * 1000)

    logging.info("Start optimizing")

    status = solver.Solve()

    output = {}
    if status == pywraplp.Solver.OPTIMAL:
        for b in d['all_bins']:
            for i in d['all_items']:
                if x[i, b].solution_value() > 0:
                    output[molecules[i]] = names[b]
        return output
    else:
        logging.warning(
            'The ILP cannot be solved. Please consider a relaxed clustering, i.e., more clusters, or a higher limit.'
        )
    return None
