import logging
from typing import Dict, Tuple, Set, List, Optional

from ortools.sat.python import cp_model
from sortedcontainers import SortedList

from scala.ilp_split.sats.id_cold_single import STATUS


def solve_ic_ilp(
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
    model = cp_model.CpModel()
    if model is None:
        print(f"GLOP solver not available.")
        return None

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

    for i in range(len(drugs)):
        model.Add(sum(x_d[i, b] for b in range(len(splits))) == 1)
    for j in range(len(proteins)):
        model.Add(sum(x_p[j, b] for b in range(len(splits))) == 1)
    for i, drug in enumerate(drugs):
        for j, protein in enumerate(proteins):
            if (drug, protein) in inter:
                model.Add(sum(x_e[i, j, b] for b in range(len(splits))) <= 1)

    for b in range(len(splits)):
        var = sum(x_e[i, j, b] for i in range(len(drugs)) for j in range(len(proteins)) if (drugs[i], proteins[j]) in inter)
        model.Add(0 < var)

        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    model.Add(x_e[i, j, b] == 1).OnlyEnforceIf(x_d[i, b]).OnlyEnforceIf(x_p[j, b])

                    model.Add(x_d[i, b] == 1).OnlyEnforceIf(x_e[i, j, b])
                    model.Add(x_p[j, b] == 1).OnlyEnforceIf(x_e[i, j, b])

    model.Maximize(
        sum(x_e[i, j, b] for i in range(len(drugs)) for j in range(len(proteins)) for b in range(len(splits)) if (drugs[i], proteins[j]) in inter)
    )

    solver = cp_model.CpSolver()

    logging.info("Start optimizing")
    status = solver.Solve(model)

    print(STATUS[status])
    print("Blube")

    output = ([], {}, {})
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i, drug in enumerate(drugs):
            for b in range(len(splits)):
                if solver.Value(x_d[i, b]) > 0:
                    print(drugs[i], names[b])
        for j, protein in enumerate(proteins):
            for b in range(len(splits)):
                if solver.Value(x_p[j, b]) > 0:
                    print(proteins[j], names[b])
        for i, drug in enumerate(drugs):
            for j, protein in enumerate(proteins):
                if (drug, protein) in inter:
                    for b in range(len(splits)):
                        if solver.Value(x_e[i, j, b]) > 0:
                            print(drugs[i], proteins[j], names[b])
                    if sum(solver.Value(x_e[i, j, b]) for b in range(len(splits))) == 0:
                        print(drugs[i], proteins[j], "not selected")
        return output
    else:
        logging.warning(
            'The ILP cannot be solved. Please consider a relaxed clustering, i.e., more clusters, or a higher limit.'
        )
    return None


def manually():
    model = cp_model.CpModel()
    x_d = {
        0: model.NewIntVar(0, 2, "x_d_0"),
        1: model.NewIntVar(0, 2, "x_d_1"),
        2: model.NewIntVar(0, 2, "x_d_2"),
        3: model.NewIntVar(0, 2, "x_d_3"),
    }
    x_p = {
        0: model.NewIntVar(0, 2, "x_p_0"),
        1: model.NewIntVar(0, 2, "x_p_1"),
        2: model.NewIntVar(0, 2, "x_p_2"),
        3: model.NewIntVar(0, 2, "x_p_3"),
    }
    x_e = {
        (0, 0): model.NewIntVar(-1, 2, "x_e_0_0"),
        (0, 1): model.NewIntVar(-1, 2, "x_e_0_1"),
        (1, 0): model.NewIntVar(-1, 2, "x_e_1_0"),
        (1, 1): model.NewIntVar(-1, 2, "x_e_1_1"),
        (2, 2): model.NewIntVar(-1, 2, "x_e_2_2"),
        (2, 3): model.NewIntVar(-1, 2, "x_e_2_3"),
        (3, 2): model.NewIntVar(-1, 2, "x_e_3_2"),
        (3, 3): model.NewIntVar(-1, 2, "x_e_3_3"),
    }
    model.Add(0 < sum([
        x_e[0, 0] == 0, x_e[0, 1] == 0, x_e[1, 0] == 0, x_e[1, 1] == 0,
        x_e[2, 2] == 0, x_e[2, 3] == 0, x_e[3, 2] == 0, x_e[3, 3] == 0,
    ]))
    model.Add(0 < sum([
        x_e[0, 0] == 1, x_e[0, 1] == 1, x_e[1, 0] == 1, x_e[1, 1] == 1,
        x_e[2, 2] == 1, x_e[2, 3] == 1, x_e[3, 2] == 1, x_e[3, 3] == 1,
    ]))

    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    print(STATUS[status])


if __name__ == '__main__':
    if False:
        manually()
    else:
        solve_ic_ilp(
            SortedList(["D1", "D2", "D3", "D4"]),
            {"D1": 1, "D2": 1, "D3": 1, "D4": 1},
            SortedList(["P1", "P2", "P3", "P4"]),
            {"P1": 1, "P2": 1, "P3": 1, "P4": 1},
            {
                ("D1", "P1"), ("D1", "P2"), ("D2", "P1"), ("D2", "P2"),
                ("D3", "P3"), ("D3", "P4"), ("D4", "P3"), ("D4", "P4"),
            },
            1,
            [0.5, 0.5],
            ["S1", "S2"],
            -1,
            -1,
        )
