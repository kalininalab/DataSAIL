import numpy as np

from datasail.solver.blp.id_cold_single import solve_ics_blp
from datasail.solver.blp.id_cold_double import solve_icd_blp
from datasail.solver.blp.cluster_cold_single import solve_ccs_blp
from datasail.solver.blp.cluster_cold_double import solve_ccd_blp


def test_ics():
    solution = solve_ics_blp(
        entities=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        weights=[6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
        epsilon=0.05,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )
    assert isinstance(solution, dict)
    weights = {"train": 0, "test": 0}
    for i in range(1, 11):
        weights[solution[f"D{i}"]] += 6 if i <= 6 else 4
    assert weights["train"] > 33.8
    assert weights["test"] > 13


def test_icd():
    solution = solve_icd_blp(
        e_entities=["D1", "D2", "D3", "D4", "D5"],
        f_entities=["P1", "P2", "P3", "P4", "P5"],
        inter={
            ("D1", "P1"), ("D1", "P2"), ("D1", "P3"),
            ("D2", "P1"), ("D2", "P2"), ("D2", "P3"),
            ("D3", "P1"), ("D3", "P2"), ("D3", "P3"),
            ("D4", "P4"), ("D4", "P5"),
            ("D5", "P4"), ("D5", "P5"),
        },
        epsilon=0.01,
        splits=[0.69, 0.31],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )
    assert isinstance(solution, tuple)
    assert all([isinstance(solution[i], dict) for i in range(3)])
    i_sol, e_sol, f_sol = solution
    for i in range(1, 4):
        assert e_sol[f"D{i}"] == "train"
        assert f_sol[f"P{i}"] == "train"
    for i in range(4, 6):
        assert e_sol[f"D{i}"] == "test"
        assert f_sol[f"P{i}"] == "test"
    for i in range(1, 4):
        for j in range(1, 4):
            assert i_sol[f"D{i}", f"P{j}"] == "train"
    for i in range(4, 6):
        for j in range(4, 6):
            assert i_sol[f"D{i}", f"P{j}"] == "test"


def test_ccd():
    solution = solve_ccd_blp(
        e_clusters=["D1", "D2", "D3"],
        e_similarities=np.asarray([
            [5, 5, 0],
            [5, 5, 0],
            [0, 0, 5],
        ]),
        e_distances=None,
        f_clusters=["P1", "P2", "P3"],
        f_similarities=np.asarray([
            [5, 5, 0],
            [5, 5, 0],
            [0, 0, 5],
        ]),
        f_distances=None,
        inter=np.asarray([
            [9, 9, 0],
            [9, 9, 0],
            [0, 0, 9],
        ]),
        epsilon=0.05,
        splits=[0.8, 0.2],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )
    assert isinstance(solution, tuple)
    assert all([isinstance(solution[i], dict) for i in range(3)])
    i_sol, e_sol, f_sol = solution
    assert e_sol["D1"] == "train"
    assert e_sol["D2"] == "train"
    assert e_sol["D3"] == "test"
    assert f_sol["P1"] == "train"
    assert f_sol["P2"] == "train"
    assert f_sol["P3"] == "test"
    assert i_sol["D1", "P1"] == "train"
    assert i_sol["D2", "P1"] == "train"
    assert i_sol["D1", "P2"] == "train"
    assert i_sol["D2", "P2"] == "train"
    assert i_sol["D3", "P3"] == "test"


def test_ccs_sim():
    solution = solve_ccs_blp(
        clusters=["1", "2", "3", "4", "5"],
        weights=[3, 3, 3, 2, 2],
        similarities=np.asarray([
            [1.0, 1.0, 1.0, 0.2, 0.2],
            [1.0, 1.0, 1.0, 0.2, 0.2],
            [1.0, 1.0, 1.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 1.0, 1.0],
            [0.2, 0.2, 0.2, 1.0, 1.0],
        ]),
        distances=None,
        epsilon=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )
    assert isinstance(solution, dict)
    for i in range(1, 6):
        assert solution[str(i)] == "train" if i < 4 else "test"


def test_ccs_sim_3c():
    solution = solve_ccs_blp(
        clusters=["1", "2", "3", "4", "5"],
        weights=[30, 30, 50, 20, 20],
        similarities=np.asarray([
            [1.0, 1.0, 0.2, 0.2, 0.2],
            [1.0, 1.0, 0.2, 0.2, 0.2],
            [0.2, 0.2, 1.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 1.0, 1.0],
            [0.2, 0.2, 0.2, 1.0, 1.0],
        ]),
        distances=None,
        epsilon=0.05,
        splits=[0.4, 0.33, 0.27],
        names=["train", "val", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )
    assert isinstance(solution, dict)
    assert solution["1"] == "train"
    assert solution["2"] == "train"
    assert solution["3"] == "val"
    assert solution["4"] == "test"
    assert solution["5"] == "test"


def test_ccs_dist():
    solution = solve_ccs_blp(
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
        epsilon=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )
    assert isinstance(solution, dict)
    for i in range(1, 6):
        assert solution[str(i)] == "train" if i < 4 else "test"
