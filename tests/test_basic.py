import numpy as np

from datasail.solver.bqp.id_cold_single import solve_ics_bqp
from datasail.solver.bqp.id_cold_double import solve_icd_bqp
from datasail.solver.bqp.cluster_cold_single import solve_ccs_bqp
from datasail.solver.bqp.cluster_cold_double import solve_ccd_bqp


def test_ics():
    assert solve_ics_bqp(
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


def test_icd():
    assert solve_icd_bqp(
        e_entities=["D1", "D2", "D3", "D4", "D5"],
        f_entities=["P1", "P2", "P3", "P4", "P5"],
        inter={
            ("D1", "P1"), ("D1", "P2"), ("D1", "P3"),
            ("D2", "P1"), ("D2", "P2"), ("D2", "P3"),
            ("D3", "P1"), ("D3", "P2"), ("D3", "P3"),
            ("D4", "P4"), ("D4", "P5"),
            ("D5", "P4"), ("D5", "P5"),
        },
        epsilon=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    )


def test_ccd():
    assert solve_ccd_bqp(
        ["D1", "D2", "D3"],
        [18, 18, 9],
        np.asarray([
            [5, 5, 0],
            [5, 5, 0],
            [0, 0, 5],
        ]),
        None,
        4,
        ["P1", "P2", "P3"],
        [18, 18, 9],
        np.asarray([
            [5, 5, 0],
            [5, 5, 0],
            [0, 0, 5],
        ]),
        None,
        4,
        np.asarray([
            [9, 9, 0],
            [9, 9, 0],
            [0, 0, 9],
        ]),
        0.2,
        [0.8, 0.2],
        ["train", "test"],
        10,
        0,
        solver="SCIP",
        log_file="./solver.log",
    ) is not None


def test_ccs_sim():
    assert solve_ccs_bqp(
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
        threshold=1,
        epsilon=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    ) is not None


def test_ccs_sim_3c():
    assert solve_ccs_bqp(
        clusters=["1", "2", "3", "4", "5"],
        weights=[3, 3, 5, 2, 2],
        similarities=np.asarray([
            [1.0, 1.0, 0.2, 0.2, 0.2],
            [1.0, 1.0, 0.2, 0.2, 0.2],
            [0.2, 0.2, 1.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 1.0, 1.0],
            [0.2, 0.2, 0.2, 1.0, 1.0],
        ]),
        distances=None,
        threshold=1,
        epsilon=0.2,
        splits=[0.4, 0.33, 0.27],
        names=["train", "val", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    ) is not None


def test_ccs_dist():
    assert solve_ccs_bqp(
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
        threshold=1,
        epsilon=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
        solver="SCIP",
        log_file="./solver.log",
    ) is not None
