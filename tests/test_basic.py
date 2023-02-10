import numpy as np
import pytest

from datasail.solver.scalar.id_cold_single import solve_ics_bqp as solve_ics_bqp_scalar
from datasail.solver.vector.id_cold_single import solve_ics_bqp as solve_ics_bqp_vector
from datasail.solver.scalar.id_cold_double import solve_icd_bqp as solve_icd_bqp_scalar
from datasail.solver.vector.id_cold_double import solve_icd_bqp as solve_icd_bqp_vector
from datasail.solver.scalar.cluster_cold_single import solve_ccs_bqp as solve_ccs_bqp_scalar
from datasail.solver.vector.cluster_cold_single import solve_ccs_bqp as solve_ccs_bqp_vector
from datasail.solver.scalar.cluster_cold_double import solve_ccd_bqp as solve_ccd_bqp_scalar
from datasail.solver.vector.cluster_cold_double import solve_ccd_bqp as solve_ccd_bqp_vector


def test_ics_scalar():
    assert solve_ics_bqp_scalar(
        e_entities=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        e_weights=[6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None


def test_icd_scalar():
    assert solve_icd_bqp_scalar(
        e_entities=["D1", "D2", "D3", "D4", "D5"],
        f_entities=["P1", "P2", "P3", "P4", "P5"],
        inter={
            ("D1", "P1"), ("D1", "P2"), ("D1", "P3"),
            ("D2", "P1"), ("D2", "P2"), ("D2", "P3"),
            ("D3", "P1"), ("D3", "P2"), ("D3", "P3"),
            ("D4", "P4"), ("D4", "P5"),
            ("D5", "P4"), ("D5", "P5"),
        },
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None


@pytest.mark.group
def test_ccs_scalar():
    test_ccs_sim_scalar()
    test_ccs_dist_scalar()


def test_ccd_scalar():
    assert solve_ccd_bqp_scalar(
        ["D1", "D2", "D3"],
        np.asarray([
            [5, 5, 0],
            [5, 5, 0],
            [0, 0, 5],
        ]),
        None,
        4,
        ["P1", "P2", "P3"],
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
    ) is not None


def test_ics_vector():
    assert solve_ics_bqp_vector(
        e_entities=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        e_weights=[6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    )


def test_icd_vector():
    assert solve_icd_bqp_vector(
        e_entities=["D1", "D2", "D3", "D4", "D5"],
        f_entities=["P1", "P2", "P3", "P4", "P5"],
        inter={
            ("D1", "P1"), ("D1", "P2"), ("D1", "P3"),
            ("D2", "P1"), ("D2", "P2"), ("D2", "P3"),
            ("D3", "P1"), ("D3", "P2"), ("D3", "P3"),
            ("D4", "P4"), ("D4", "P5"),
            ("D5", "P4"), ("D5", "P5"),
        },
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    )


@pytest.mark.group
def test_ccs_vector():
    test_ccs_sim_vector()
    test_ccs_dist_vector()


def test_ccd_vector():
    assert solve_ccd_bqp_vector(
        ["D1", "D2", "D3"],
        np.asarray([
            [5, 5, 0],
            [5, 5, 0],
            [0, 0, 5],
        ]),
        None,
        4,
        ["P1", "P2", "P3"],
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
    ) is not None


@pytest.mark.todo
@pytest.mark.parametrize("size", list(range(5, 51, 5)))
def test_ccs_sim_scalar_nw(size):
    assert solve_ccs_bqp_scalar(
        e_clusters=[str(i + 1) for i in range(size)],
        e_weights=[1] * size,
        e_similarities=np.ones((size, size)),
        e_distances=None,
        e_threshold=1,
        limit=0.25,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None


def test_ccs_sim_scalar():
    assert solve_ccs_bqp_scalar(
        e_clusters=["1", "2", "3", "4", "5"],
        e_weights=[3, 3, 3, 2, 2],
        e_similarities=np.asarray([
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ]),
        e_distances=None,
        e_threshold=1,
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None


def test_ccs_dist_scalar():
    assert solve_ccs_bqp_scalar(
        e_clusters=["1", "2", "3", "4", "5"],
        e_weights=[3, 3, 3, 2, 2],
        e_similarities=None,
        e_distances=np.asarray([
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [4, 4, 4, 0, 0],
            [4, 4, 4, 0, 0],
        ]),
        e_threshold=1,
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None


def test_ccs_sim_vector():
    assert solve_ccs_bqp_vector(
        e_clusters=["1", "2", "3", "4", "5"],
        e_weights=[3, 3, 3, 2, 2],
        e_similarities=np.asarray([
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ]),
        e_distances=None,
        e_threshold=1,
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None


def test_ccs_dist_vector():
    assert solve_ccs_bqp_vector(
        e_clusters=["1", "2", "3", "4", "5"],
        e_weights=[3, 3, 3, 2, 2],
        e_similarities=None,
        e_distances=np.asarray([
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [4, 4, 4, 0, 0],
            [4, 4, 4, 0, 0],
        ]),
        e_threshold=1,
        limit=0.2,
        splits=[0.7, 0.3],
        names=["train", "test"],
        max_sec=10,
        max_sol=0,
    ) is not None
