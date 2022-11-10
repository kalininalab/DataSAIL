import os

import numpy as np

from scala.ilp_split.read_data import read_data


def ilp_main(args):
    data = read_data(args)
    output = {"inter": None, "drug": None, "prot": None}

    if args.technique == "R":
        output["inter"] = list(sample_categorical(data["interactions"], args.splits))
    if args.technique == "ICD":
        pass
    if args.technique == "ICP":
        pass
    if args.technique == "IC":
        pass
    if args.technique == "CCD":
        pass
    if args.technique == "CCP":
        pass
    if args.technique == "CC":
        pass

    if output["inter"] is not None:
        with open(os.path.join(args.output, "inter.tsv"), "w") as stream:
            for s, split in enumerate(output["inter"]):
                for drug, prot in split:
                    print(drug, prot, args.names[s], sep=args.sep, file=stream)
    if output["drug"] is not None:
        with open(os.path.join(args.output, "drug.tsv"), "w") as stream:
            for s, split in enumerate(output["drug"]):
                for drug in split:
                    print(drug, args.names[s], sep=args.sep, file=stream)
    if output["prot"] is not None:
        with open(os.path.join(args.output, "prot.tsv"), "w") as stream:
            for s, split in enumerate(output["prot"]):
                for prot in split:
                    print(prot, args.names[s], sep=args.sep, file=stream)


def sample_categorical(data, splits):
    np.random.shuffle(data)
    for i in range(len(splits) - 1):
        yield data[int(sum(splits[:i]) * len(data)):int(sum(splits[:(i + 1)]) * len(data))]
    yield data[int(sum(splits[:-1]) * len(data)):]


def solve_mpk_ilp_icd(drugs, weights, limit, splits):
    np.random.shuffle(drugs)
    d = {
        "weights": weights,
        "values": weights,
        "num_items": len(drugs),
        "all_items": range(len(drugs)),
        "bin_capacities": [s * sum(weights) * (1 + limit) for s in splits],
        "num_bins": len(splits),
        "all_bins": range(len(splits)),
    }

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        print('SCIP solver unavailable.')
        return

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
        solver.Add(
            sum(x[i, b] * d['weights'][i]
                for i in d['all_items']) <= d['bin_capacities'][b])

    # Objective.
    # Maximize total value of packed items.
    objective = solver.Objective()
    for i in d['all_items']:
        for b in d['all_bins']:
            objective.SetCoefficient(x[i, b], d['values'][i])
    objective.SetMaximization()

    status = solver.Solve()

    bins = [[], [], []]
    if status == pywraplp.Solver.OPTIMAL:
        for b in d['all_bins']:
            for i in d['all_items']:
                if x[i, b].solution_value() > 0:
                    bins[b] += drugs[i]
        return bins
    else:
        print('The problem does not have an optimal solution.')
    return None, None, None
