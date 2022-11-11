import logging
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from ortools.linear_solver import pywraplp
from sortedcontainers import SortedList

from scala.ilp_split.read_data import read_data


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
        )
    if args.technique == "ICP":
        prot = SortedList(data["proteins"].keys())
        output["proteins"] = solve_mpk_ilp_icx(
            prot,
            [data["prot_weights"][p] for p in prot],
            args.limit,
            args.splits,
            args.names,
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


def solve_mpk_ilp_icx(
        molecules: SortedList,
        weights: List[float],
        limit: float,
        splits: List[float],
        names: List[str],
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

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        print('SCIP solver unavailable.')
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


def solve_mpk_ilp_ic(
        drugs: SortedList,
        drug_weights: Dict[str, float],
        proteins: SortedList,
        protein_weights: Dict[str, float],
        inter: Set[Tuple[str, str]],
        limit: float,
        splits: List[float],
        names: List[str],
) -> Optional[List[List[Tuple[str, str, str]], Dict[str, str], Dict[str, str]]]:
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        print('SCIP solver unavailable.')
        return None

    # Variables.
    # x[i, b] = 1 if item i is packed in bin b.
    x_d = {}
    for i in range(len(drugs)):
        for b in range(len(splits)):
            x_d[i, b] = solver.BoolVar(f'x_d_{i}_{b}')
    x_p = {}
    for j in range(len(proteins)):
        for b in range(len(splits)):
            x_p[j, b] = solver.BoolVar(f'x_p_{j}_{b}')
    x_e = {}
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                x_e[i, j] = solver.BoolVar(f'x_e_{i}_{j}')

    for i in range(len(drugs)):
        solver.Add(sum(x_d[i, b] for b in range(len(splits))) <= 1)
    for j in range(len(proteins)):
        solver.Add(sum(x_p[j, b] for b in range(len(splits))) <= 1)

    for b in range(len(splits)):
        solver.Add(
            sum(x_d[i, b] * drug_weights[drugs[i]]
                for i in range(len(drugs))) <= splits[b] * len(inter) * (1 + limit)
        )
        solver.Add(
            sum(x_p[j, b] * protein_weights[proteins[j]]
                for j in range(len(proteins))) <= splits[b] * len(inter) * (1 + limit)
        )

    for b in range(len(splits)):
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    solver.Add(x_e[i, j] <= 0.75 * (x_d[i, b] + x_p[j, b]))

    # Objective. Maximize total value of packed items.
    objective = solver.Objective()
    for i in range(len(drugs)):
        for b in range(len(splits)):
            objective.SetCoefficient(x_d[i, b], 1)
    for j in range(len(proteins)):
        for b in range(len(splits)):
            objective.SetCoefficient(x_p[j, b], 1)
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                objective.SetCoefficient(x_e[i, j], 1)
    objective.SetMaximization()

    logging.info("Start optimization")

    status = solver.Solve()

    output = [[], {}, {}]
    if status == pywraplp.Solver.OPTIMAL:
        for i, drug in enumerate(drugs):
            for b in range(len(splits)):
                if x_d[i, b].solution_value() > 0:
                    output[1][drug] = names[b]
            if drug not in output[1]:
                output[1][drug] = "not selected"
        for j, protein in enumerate(proteins):
            for b in range(len(splits)):
                if x_p[j, b].solution_value() > 0:
                    output[2][protein] = names[b]
            if protein not in output[2]:
                output[2][protein] = "not selected"
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    if x_e[i, j].solution_value() > 0:
                        output[0].append((drug, protein, output[1][drug]))
                    else:
                        output[0].append((drug, protein, "not selected"))
        return output
    else:
        logging.warning(
            'The ILP cannot be solved. Please consider a relaxed clustering, i.e., more clusters, or a higher limit.'
        )
    return None
