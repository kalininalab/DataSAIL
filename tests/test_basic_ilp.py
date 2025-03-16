from pathlib import Path

import numpy as np

from datasail.solver.id_1d import solve_i1
from datasail.solver.id_2d import solve_i2
from datasail.solver.cluster_1d import solve_c1
from datasail.solver.cluster_2d import solve_c2


def test_ics():
    solution = solve_i1(
        entities=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        weights=[6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
        stratification=None,
        epsilon=0.05,
        delta=0.05,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file=Path("solver.log"),
    )
    assert isinstance(solution, dict)
    weights = {"train": 0, "test": 0}
    for i in range(1, 11):
        weights[solution[f"D{i}"]] += 6 if i <= 6 else 4
    assert weights["train"] > 33.8
    assert weights["test"] > 13


def test_icd():
    solution = solve_i2(
        e_entities=["D1", "D2", "D3", "D4", "D5"],
        e_stratification=None,
        e_weights=np.array([1, 1, 1, 1, 1]),
        f_entities=["P1", "P2", "P3", "P4", "P5"],
        f_stratification=None,
        f_weights=np.array([1, 1, 1, 1, 1]),
        inter={
            ("D1", "P1"), ("D1", "P2"), ("D1", "P3"),
            ("D2", "P1"), ("D2", "P2"), ("D2", "P3"),
            ("D3", "P1"), ("D3", "P2"), ("D3", "P3"),
            ("D4", "P4"), ("D4", "P5"),
            ("D5", "P4"), ("D5", "P5"),
        },
        epsilon=0.01,
        delta=0.01,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file=Path("solver.log"),
    )
    assert isinstance(solution, tuple)
    assert all([isinstance(solution[i], dict) for i in range(3)])
    i_sol, e_sol, f_sol = solution
    for i in range(1, 6):
        assert e_sol[f"D{i}"] in {"train", "test"}
        assert f_sol[f"P{i}"] in {"train", "test"}
    selections = {"train": 0, "test": 0}
    for i in range(1, 4):
        for j in range(1, 4):
            assert i_sol[f"D{i}", f"P{j}"] in {"train", "test", "not selected"}
            if i_sol[f"D{i}", f"P{j}"] != "not selected":
                selections[i_sol[f"D{i}", f"P{j}"]] += 1
    for i in range(4, 6):
        for j in range(4, 6):
            assert i_sol[f"D{i}", f"P{j}"] in {"train", "test", "not selected"}
            if i_sol[f"D{i}", f"P{j}"] != "not selected":
                selections[i_sol[f"D{i}", f"P{j}"]] += 1
    print(selections)
    assert selections["train"] >= int(0.7 * 0.99 * (selections["train"] + selections["test"]))


def test_ccd():
    solution = solve_c2(
        e_clusters=["D1", "D2", "D3", "D4", "D5"],
        e_similarities=np.asarray([
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ]),
        e_distances=None,
        e_s_matrix=None,
        e_weights=np.array([3, 3, 3, 3, 3]),
        f_clusters=["P1", "P2", "P3", "P4", "P5"],
        f_similarities=np.asarray([
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ]),
        f_distances=None,
        f_s_matrix=None,
        f_weights=np.array([3, 3, 3, 3, 3]),
        inter=np.asarray([
            [9, 9, 9, 0, 0],
            [9, 9, 9, 0, 0],
            [9, 9, 9, 0, 0],
            [0, 0, 0, 9, 9],
            [0, 0, 0, 9, 9],
        ]),
        epsilon=0.05,
        delta=0.05,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file=Path("solver.log"),
    )
    assert isinstance(solution, tuple)
    assert all([isinstance(solution[i], dict) for i in range(3)])
    i_sol, e_sol, f_sol = solution
    for i in range(1, 6):
        assert e_sol[f"D{i}"] in {"train", "test"}
        assert f_sol[f"P{i}"] in {"train", "test"}
    selections = {"train": 0, "test": 0}
    for i in range(1, 4):
        for j in range(1, 4):
            assert i_sol[f"D{i}", f"P{j}"] in {"train", "test", "not selected"}
            if i_sol[f"D{i}", f"P{j}"] != "not selected":
                selections[i_sol[f"D{i}", f"P{j}"]] += 1
    for i in range(4, 6):
        for j in range(4, 6):
            assert i_sol[f"D{i}", f"P{j}"] in {"train", "test", "not selected"}
            if i_sol[f"D{i}", f"P{j}"] != "not selected":
                selections[i_sol[f"D{i}", f"P{j}"]] += 1
    print(selections)
    assert selections["train"] >= int(0.7 * 0.99 * (selections["train"] + selections["test"]))


def test_ccs_sim():
    solution = solve_c1(
        clusters=["1", "2", "3", "4", "5"],
        weights=[3, 3, 3, 2, 2],
        similarities=np.asarray([
            [1.0, 1.0, 1.0, 0.2, 0.2],
            [1.0, 1.0, 1.0, 0.2, 0.2],
            [1.0, 1.0, 1.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 1.0, 1.0],
            [0.2, 0.2, 0.2, 1.0, 1.0],
        ]),
        s_matrix=None,
        distances=None,
        epsilon=0.2,
        delta=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file=Path("solver.log"),
    )
    assert isinstance(solution, dict)
    for i in range(1, 6):
        assert solution[str(i)] == "train" if i < 4 else "test"


def test_ccs_sim_3c():
    solution = solve_c1(
        clusters=["1", "2", "3", "4", "5"],
        weights=[30, 30, 50, 20, 20],
        similarities=np.asarray([
            [1.0, 1.0, 0.2, 0.2, 0.2],
            [1.0, 1.0, 0.2, 0.2, 0.2],
            [0.2, 0.2, 1.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 1.0, 1.0],
            [0.2, 0.2, 0.2, 1.0, 1.0],
        ]),
        s_matrix=None,
        distances=None,
        epsilon=0.05,
        delta=0.05,
        splits=[0.4, 0.33, 0.27],
        names=["train", "val", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file=Path("solver.log"),
    )
    assert isinstance(solution, dict)
    assert solution["1"] == "train"
    assert solution["2"] == "train"
    assert solution["3"] == "val"
    assert solution["4"] == "test"
    assert solution["5"] == "test"


def test_ccs_dist():
    solution = solve_c1(
        clusters=["1", "2", "3", "4", "5"],
        weights=[3, 3, 3, 2, 2],
        similarities=None,
        distances=np.asarray([
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [4, 4, 4, 0, 0],
            [4, 4, 4, 0, 0],
        ]),
        s_matrix=None,
        epsilon=0.2,
        delta=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file=Path("solver.log"),
    )
    assert isinstance(solution, dict)
    for i in range(1, 6):
        assert solution[str(i)] == "train" if i < 4 else "test"
