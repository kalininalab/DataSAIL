import logging
from typing import List, Dict, Optional

from ortools.sat.python import cp_model
from sortedcontainers import SortedList


STATUS = {
    cp_model.OPTIMAL: "Optimal",
    cp_model.FEASIBLE: "Feasible",
    cp_model.INFEASIBLE: "Infeasible",
    cp_model.MODEL_INVALID: "Invalid model",
    cp_model.UNKNOWN: "Unknown",
}


class SolutionTracker(cp_model.CpSolverSolutionCallback):
    def __init__(self, sol_max, variables, weights, splits):
        super(SolutionTracker, self).__init__()
        self.solution = None
        self.best_score = float('inf')

        self.sol_max = sol_max
        self.sol_count = 0

        self.variables = variables
        self.weights = weights
        self.splits = splits

    def on_solution_callback(self):
        self.sol_count += 1

        score = sum((sum(self.Value(self.variables[i, b])
                         for i in range(len(self.weights))) - self.splits[b]) ** 2 for b in range(len(self.splits)))
        # print(f"{score:.5f}", [self.Value(self.variables[i, b])
        #                        for i in range(len(self.weights)) for b in range(len(self.splits))], end="")

        if score < self.best_score:
            print(" <=")
            self.best_score = score
            self.solution = dict((self.variables[i, b], self.Value(self.variables[i, b]))
                                 for i in range(len(self.weights)) for b in range(len(self.splits)))
        else:
            print()

        if 0 <= self.sol_max <= self.sol_count:
            self.StopSearch()

    def solution_count(self):
        return self.sol_count

    def get_value(self, v):
        return self.solution[v]


def solve_icx_sat(
        molecules: SortedList,
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
        max_sec: int,
        max_sol: int,
) -> Optional[Dict[str, str]]:
    # Create the mip solver with the algorithm backend.
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
        model.Add(sum(x[i, b] * weights[i] for i in range(len(molecules))) <= int(splits[b] * sum(weights) * limit))

    solver = cp_model.CpSolver()
    if max_sec != -1:
        solver.parameters.max_time_in_seconds = max_sec
    sol_tracker = SolutionTracker(
        max_sol,
        x,
        weights,
        splits,
    )
    solver.parameters.enumerate_all_solutions = True

    logging.info("Start optimizing")

    status = solver.SearchForAllSolutions(model, sol_tracker)
    # print(sol_tracker.solution)

    output = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logging.info(STATUS[status])
        for i in range(len(molecules)):
            for b in range(len(splits)):
                if sol_tracker.solution[x[i, b]] > 0:
                    output[molecules[i]] = names[b]
            if molecules[i] not in output:
                molecules[i] = "not selected"
        return output
    else:
        logging.warning(STATUS[status])
        logging.warning(
            'The ILP cannot be solved. Please consider a relaxed clustering, i.e., more clusters, or a higher limit.'
        )
    return None
