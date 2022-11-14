import logging
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from sortedcontainers import SortedList

from scala.ilp_split.ilps.id_cold_double import solve_mpk_ilp_ic
from scala.ilp_split.ilps.id_cold_single import solve_mpk_ilp_icx
from scala.ilp_split.read_data import read_data


ALGORITHM = "CP_SAT"


class MaxSolutionTerminator(cp_model.CpSolverSolutionCallback):
    def __init__(self, max_sol):
        super(MaxSolutionTerminator, self).__init__()
        self.__sol_count = 0
        self.__max_num_sol = max_sol

    def on_solution_callback(self):
        self.__sol_count += 1
        if self.__sol_count >= self.__max_num_sol:
            self.StopSearch()

    def solution_count(self):
        return self.__sol_count


class SATObjectiveCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, max_sol, objective, variables):
        super(SATObjectiveCallback, self).__init__()

        self.__sol_count = 0
        self.__max_num_sol = max_sol

        self.__objective = objective
        self.__variables = variables
        self.best_score = 0
        # self.best_sol = 0

    def on_solution_callback(self):
        self.__sol_count += 1

        score = self.__objective(self.Value(v) for v in self.__variables)
        if score > self.best_score:
            self.best_score = score

        if self.__sol_count >= self.__max_num_sol:
            self.StopSearch()


def ilp_main(args):
    logging.info("Starting ILP solving")
    logging.info("Read data")

    data = read_data(args)
    output = {"inter": None, "drugs": None, "proteins": None}

    logging.info("Split data")

    if args.technique == "R":
        output["inter"] = sample_categorical(
            list(data["interactions"]),
            args.splits,
            args.names
        )
    if args.technique == "ICD":
        drug = SortedList(data["drugs"].keys())
        output["drugs"] = solve_mpk_ilp_icx(
            drug,
            [data["drug_weights"][d] for d in drug],
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
    if args.technique == "ICP":
        prot = SortedList(data["proteins"].keys())
        output["proteins"] = solve_mpk_ilp_icx(
            prot,
            [data["prot_weights"][p] for p in prot],
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
    if args.technique == "IC":
        solution = solve_mpk_ilp_ic(
            SortedList(data["drugs"].keys()),
            data["drug_weights"],
            SortedList(data["proteins"].keys()),
            data["prot_weights"],
            set(tuple(x) for x in data["interactions"]),
            args.limit,
            args.splits,
            args.names,
            args.max_sec,
            args.max_sol,
        )
        if solution is not None:
            output["inter"], output["drugs"], output["proteins"] = solution
    if args.technique == "CCD":
        pass
    if args.technique == "CCP":
        pass
    if args.technique == "CC":
        pass

    logging.info("Store results")

    if output["inter"] is None and output["drugs"] is not None and output["proteins"] is None:
        output["inter"] = [(d, p, output["drugs"][d]) for d, p in data["interactions"]]
    if output["inter"] is None and output["drugs"] is None and output["proteins"] is not None:
        output["inter"] = [(d, p, output["proteins"][p]) for d, p in data["interactions"]]

    if output["inter"] is not None:
        split_stats = dict((n, 0) for n in args.names + ["not selected"])
        with open(os.path.join(args.output, "inter.tsv"), "w") as stream:
            for drug, prot, split in output["inter"]:
                print(drug, prot, split, sep=args.sep, file=stream)
                split_stats[split] += 1
        print("Interaction-split statistics:")
        print('\n'.join([f"\t{k}\t: {v} {100 * v / len(data['interactions']):.2f}%" for k, v in split_stats.items()]))

    if output["drugs"] is not None:
        split_stats = dict((n, 0) for n in args.names + ["not selected"])
        with open(os.path.join(args.output, "drug.tsv"), "w") as stream:
            for drug, split in output["drugs"].items():
                print(drug, split, sep=args.sep, file=stream)
                split_stats[split] += 1
        print("Drug distribution over splits:")
        print('\n'.join([f"\t{k}\t: {v} {100 * v / len(data['drugs']):.2f}%" for k, v in split_stats.items()]))

    if output["proteins"] is not None:
        split_stats = dict((n, 0) for n in args.names + ["not selected"])
        with open(os.path.join(args.output, "prot.tsv"), "w") as stream:
            for protein, split in output["proteins"].items():
                print(protein, split, sep=args.sep, file=stream)
                split_stats[split] += 1
        print("Protein distribution over splits:")
        print('\n'.join([f"\t{k}\t: {v} {100 * v / len(data['proteins']):.2f}%" for k, v in split_stats.items()]))

    logging.info("ILP splitting finished and results stored.")


def sample_categorical(
        data: List[Tuple[str, str]],
        splits: List[float],
        names: List[str],
):
    np.random.shuffle(data)

    def gen():
        for index in range(len(splits) - 1):
            yield data[int(sum(splits[:index]) * len(data)):int(sum(splits[:(index + 1)]) * len(data))]
        yield data[int(sum(splits[:-1]) * len(data)):]

    output = []
    for i, split in enumerate(gen()):
        output += [(d, p, names[i]) for d, p in split]
    return output
