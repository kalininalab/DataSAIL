import logging

from ortools.sat.python import cp_model

STATUS = {
    cp_model.OPTIMAL: "Optimal",
    cp_model.FEASIBLE: "Feasible",
    cp_model.INFEASIBLE: "Infeasible",
    cp_model.MODEL_INVALID: "Invalid model",
    cp_model.UNKNOWN: "Unknown",
}


class SolutionTracker(cp_model.CpSolverSolutionCallback):
    def __init__(self, max_num_sols):
        super(SolutionTracker, self).__init__()
        self.__max_num_sols = max_num_sols
        self.__sol_count = 0

    def on_solution_callback(self):
        self.__sol_count += 1
        if self.__sol_count >= self.__max_num_sols:
            logging.info("Stop search by maximal number of found solutions.")
            self.StopSearch()
