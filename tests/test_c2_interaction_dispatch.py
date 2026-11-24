"""C2 dispatch uses cluster-level interaction counts in cluster row order."""
import math

import cvxpy as cp
import numpy as np
import pytest

from datasail.reader.utils import DataSet
from datasail.sail import datasail
from datasail.solver.cluster_2d import solve_c2
from datasail.solver.solve import run_solver


INTERACTIONS = [
    ("L0", "P0"), ("L0", "P1"), ("L1", "P0"), ("L1", "P2"),
    ("L2", "P1"), ("L2", "P2"), ("L3", "P3"), ("L3", "P0"),
]
LABELS = dict(zip(INTERACTIONS, [0, 1, 0, 0, 1, 1, 0, 1]))


def incidence_counts(interactions, labels, membership, cluster_names, axis):
    """Count original interaction incidences, not partners or weighted entities."""
    rows = {cluster: i for i, cluster in enumerate(cluster_names)}
    counts = np.zeros((len(cluster_names), 2))
    for pair in interactions:
        counts[rows[membership[pair[axis]]], labels[pair]] += 1
    return counts


def isolated_dataset(axis, collision=True, identity=False):
    prefix = "L" if axis == 0 else "P"
    names = [f"{prefix}{i}" for i in range(4)]
    if identity:
        clusters = names[::-1]
        membership = {name: name for name in names}
    else:
        # Deliberately reverse cluster row order; labels may also be entity IDs.
        first = names[0] if collision else f"{prefix}-first"
        second = names[2 if axis == 0 else 3] if collision else f"{prefix}-second"
        clusters = [second, first]
        membership = {name: first if i < 2 else second for i, name in enumerate(names)}
    entity_counts = incidence_counts(INTERACTIONS, LABELS, {n: n for n in names}, names, axis)
    cluster_counts = incidence_counts(INTERACTIONS, LABELS, membership, clusters, axis)
    weights = {name: 3 * (i + 1) for i, name in enumerate(names)}
    return DataSet(
        names=names, cluster_names=clusters, cluster_map=membership, weights=weights,
        cluster_weights={c: sum(weights[n] for n in names if membership[n] == c) for c in clusters},
        cluster_similarity=np.eye(len(clusters)), interaction_classes={0: 0, 1: 1},
        interaction_stratification=dict(zip(names, entity_counts)),
        cluster_interaction_stratification=dict(zip(clusters, cluster_counts)),
    ), cluster_counts


def isolated_dispatch(monkeypatch, e, f):
    captured = []

    def inspect(**kwargs):
        captured.append(kwargs)
        return None

    monkeypatch.setattr("datasail.solver.solve.solve_c2", inspect)
    run_solver(techniques=["C2"], e_dataset=e, f_dataset=f, delta=.05, epsilon=.05, runs=1,
               split_ratios={"C2e": [.6, .4], "C2f": [.6, .4]},
               split_names={"C2e": ["train", "test"], "C2f": ["train", "test"]},
               max_sec=2, solver="SCIP", log_dir=None)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("axis", ["e", "f"])
@pytest.mark.parametrize("collision", [False, True])
def test_isolated_merged_counts_and_row_order(monkeypatch, collision, axis):
    e, expected_e = isolated_dataset(0, collision)
    f, expected_f = isolated_dataset(1, collision)
    # L0+L1 includes L0-P0, L1-P0 and L1-P2 in class 0, plus L0-P1 in class 1.
    np.testing.assert_array_equal(expected_e, [[1, 3], [3, 1]])
    np.testing.assert_array_equal(expected_f, [[2, 1], [2, 3]])
    actual = isolated_dispatch(monkeypatch, e, f)
    assert actual["e_clusters"] == e.cluster_names
    assert actual["f_clusters"] == f.cluster_names
    np.testing.assert_array_equal(actual["eis_matrix" if axis == "e" else "fis_matrix"],
                                  expected_e if axis == "e" else expected_f)


@pytest.mark.parametrize("field", ["interaction_stratification", "cluster_interaction_stratification"])
def test_isolated_enablement_uses_cluster_representation(monkeypatch, field):
    e, expected_e = isolated_dataset(0)
    f, expected_f = isolated_dataset(1)
    setattr(e, field, None)
    setattr(f, field, None)
    actual = isolated_dispatch(monkeypatch, e, f)
    if field == "interaction_stratification":
        np.testing.assert_array_equal(actual["eis_matrix"], expected_e)
        np.testing.assert_array_equal(actual["fis_matrix"], expected_f)
    else:
        assert actual["eis_matrix"] is None and actual["fis_matrix"] is None


def test_isolated_identity_clusters_are_unchanged(monkeypatch):
    e, expected_e = isolated_dataset(0, identity=True)
    f, expected_f = isolated_dataset(1, identity=True)
    actual = isolated_dispatch(monkeypatch, e, f)
    np.testing.assert_array_equal(actual["eis_matrix"], expected_e)
    np.testing.assert_array_equal(actual["fis_matrix"], expected_f)


def test_isolated_absent_interaction_stratification(monkeypatch):
    e, _ = isolated_dataset(0)
    f, _ = isolated_dataset(1)
    for dataset in (e, f):
        dataset.interaction_stratification = None
        dataset.cluster_interaction_stratification = None
    actual = isolated_dispatch(monkeypatch, e, f)
    assert actual["eis_matrix"] is None and actual["fis_matrix"] is None


def public_fixture():
    ligands = [f"L{i}" for i in range(20)]
    proteins = [f"P{i}" for i in range(20)]
    interactions = [(ligands[i], proteins[i]) for i in range(20)]
    labels = {pair: 0 for pair in interactions}
    for i in range(20):
        pair = (ligands[i], proteins[(i + 1) % 20])
        interactions.append(pair)
        labels[pair] = 1
    # Exact unit similarities within each pair, zero between pairs: ten two-member clusters.
    distances = np.array([[float(i // 2 != j // 2) for j in range(20)] for i in range(20)])
    kwargs = dict(
        techniques=["C2"], splits=[8, 1, 1], names=["train", "validation", "test"],
        e_type="O", e_data={n: str(i) for i, n in enumerate(ligands)},
        f_type="O", f_data={n: str(i) for i, n in enumerate(proteins)},
        e_sim=(ligands, 1 - distances), f_sim=(proteins, 1 - distances), e_clusters=10, f_clusters=10,
        e_weights={n: 1 for n in ligands}, f_weights={n: 1 for n in proteins},
        inter=interactions, inter_strat={pair: {label} for pair, label in labels.items()},
        epsilon=.05, delta=.05, max_sec=2, threads=1,
    )
    return kwargs, interactions, labels


@pytest.mark.parametrize("native", [False, True])
def test_public_merged_counts_and_final_targets(monkeypatch, native):
    if native and "SCIP" not in cp.installed_solvers():
        pytest.skip("SCIP is unavailable")
    kwargs, interactions, labels = public_fixture()
    expected = {}
    inspected = []
    roots = [math.sqrt(r) for r in [8, 1, 1]]
    targets = np.array(roots) / sum(roots)

    def observe_dispatch(**arguments):
        for axis, key in enumerate(("e_dataset", "f_dataset")):
            dataset = arguments[key]
            groups = [{n for n, c in dataset.cluster_map.items() if c == cluster}
                      for cluster in dataset.cluster_names]
            assert len(groups) == 10 and all(len(group) == 2 for group in groups)
            names = list(kwargs["e_data" if axis == 0 else "f_data"])
            assert {frozenset(g) for g in groups} == {frozenset(names[i:i + 2]) for i in range(0, 20, 2)}
            counts = incidence_counts(interactions, labels, dataset.cluster_map, dataset.cluster_names, axis)
            np.testing.assert_array_equal(counts, np.full((10, 2), 2))
            np.testing.assert_array_equal(counts.sum(axis=0), [20, 20])
            np.testing.assert_array_equal(np.stack([dataset.cluster_interaction_stratification[c]
                                                  for c in dataset.cluster_names]), counts)
            expected[axis] = counts
        return run_solver(**arguments)  # Forward to the real dispatch; aggregation is never replaced.

    def observe_c2(**arguments):
        np.testing.assert_array_equal(arguments["eis_matrix"], expected[0])
        np.testing.assert_array_equal(arguments["fis_matrix"], expected[1])
        np.testing.assert_allclose(arguments["e_splits"], targets, rtol=0, atol=1e-8)
        np.testing.assert_allclose(arguments["f_splits"], targets, rtol=0, atol=1e-8)
        return solve_c2(**arguments)

    def inspect_model(loss, constraints, max_sec, solver, log_file):
        if native:
            from datasail.solver.utils import solve
            problem = solve(loss, constraints, max_sec, solver, log_file)
            assert problem is not None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            assert all(np.max(c.violation()) < 1e-7 for c in constraints)
            inspected.append(problem.status)
            return problem
        from tests.test_split_ratios import Model
        model = Model(constraints)
        for variable in model.variables:
            [(counts, limits)] = model.class_bounds(variable)
            np.testing.assert_array_equal(counts, np.full((10, 2), 2))
            independent_limits = np.maximum(np.floor(targets[:, None] * .95 * [20, 20]), 1)
            np.testing.assert_array_equal(independent_limits, [[11, 11], [3, 3], [3, 3]])
            np.testing.assert_array_equal(limits, independent_limits)
        inspected.append("constructed")
        return None

    monkeypatch.setattr("datasail.routine.run_solver", observe_dispatch)
    monkeypatch.setattr("datasail.solver.solve.solve_c2", observe_c2)
    monkeypatch.setattr("datasail.solver.cluster_2d.solve", inspect_model)
    result = datasail(**kwargs)
    assert len(inspected) == 1
    if native:
        for assignments in result[:2]:
            assert len(assignments["C2"][0]) == 20


@pytest.mark.parametrize("axis", [0, 1])
def test_break_rebuilds_projected_counts_after_cluster_renumbering(axis):
    from datasail.solver.overflow import break_cluster

    dataset, _ = isolated_dataset(axis)
    original_entity_counts = {n: row.copy() for n, row in dataset.interaction_stratification.items()}
    original_membership = dataset.cluster_map.copy()
    broken = dataset.cluster_names[0]
    remaining = set(dataset.names) - {n for n, c in original_membership.items() if c == broken}
    break_cluster(dataset, broken, .4, "average")
    assert len(dataset.cluster_names) == 3
    assert len({dataset.cluster_map[n] for n in remaining}) == 1
    assert len({dataset.cluster_map[n] for n, c in original_membership.items() if c == broken}) == 2
    assert set(dataset.cluster_interaction_stratification) == set(dataset.cluster_names)
    expected = incidence_counts(INTERACTIONS, LABELS, dataset.cluster_map, dataset.cluster_names, axis)
    actual = np.stack([dataset.cluster_interaction_stratification[c] for c in dataset.cluster_names])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.sum(axis=0), [4, 4])
    for name, counts in original_entity_counts.items():
        np.testing.assert_array_equal(dataset.interaction_stratification[name], counts)


@pytest.mark.parametrize("axis,native", [(0, False), (1, False), (0, True)])
def test_public_break_keeps_counts_and_constraints_consistent(monkeypatch, axis, native):
    if native and "SCIP" not in cp.installed_solvers():
        pytest.skip("SCIP is unavailable")
    kwargs, interactions, labels = public_fixture()
    prefix = "e" if axis == 0 else "f"
    similarities = np.eye(20)
    similarities[:14, :14] = 1
    kwargs.update({f"{prefix}_sim": (list(kwargs[f"{prefix}_data"]), similarities),
                   f"{prefix}_clusters": 7, "overflow": "break"})
    expected = {}
    inspected = []
    roots = np.sqrt([8, 1, 1])
    targets = roots / roots.sum()

    def observe_dispatch(**arguments):
        for side, key in enumerate(("e_dataset", "f_dataset")):
            dataset = arguments[key]
            counts = incidence_counts(interactions, labels, dataset.cluster_map, dataset.cluster_names, side)
            actual = np.stack([dataset.cluster_interaction_stratification.get(c, np.zeros(2))
                               for c in dataset.cluster_names])
            np.testing.assert_array_equal(actual, counts)
            np.testing.assert_array_equal(actual.sum(axis=0), [20, 20])
            if side == axis:
                assert len(dataset.cluster_names) == 8
                assert sorted(sum(c == cluster for c in dataset.cluster_map.values())
                              for cluster in dataset.cluster_names) == [1] * 6 + [7, 7]
            expected[side] = counts
        return run_solver(**arguments)

    def observe_c2(**arguments):
        np.testing.assert_array_equal(arguments["eis_matrix"], expected[0])
        np.testing.assert_array_equal(arguments["fis_matrix"], expected[1])
        np.testing.assert_allclose(arguments["e_splits"], targets, rtol=0, atol=1e-8)
        np.testing.assert_allclose(arguments["f_splits"], targets, rtol=0, atol=1e-8)
        return solve_c2(**arguments)

    def inspect_model(loss, constraints, max_sec, solver, log_file):
        if native:
            from datasail.solver.utils import solve
            problem = solve(loss, constraints, max_sec, solver, log_file)
            assert problem is not None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            assert all(np.max(c.violation()) < 1e-7 for c in constraints)
            inspected.append(problem.status)
            return problem
        from tests.test_split_ratios import Model
        model = Model(constraints)
        for variable in model.variables:
            [(counts, limits)] = model.class_bounds(variable)
            expected_counts = next(c for c in expected.values() if len(c) == variable.shape[1])
            np.testing.assert_array_equal(counts, expected_counts)
            np.testing.assert_array_equal(limits, [[11, 11], [3, 3], [3, 3]])
        inspected.append("constructed")
        return None

    monkeypatch.setattr("datasail.routine.run_solver", observe_dispatch)
    monkeypatch.setattr("datasail.solver.solve.solve_c2", observe_c2)
    monkeypatch.setattr("datasail.solver.cluster_2d.solve", inspect_model)
    result = datasail(**kwargs)
    assert len(inspected) == 1
    if native:
        assert all(len(mapping["C2"][0]) == 20 for mapping in result[:2])


def entity_incidence_counts(labels, membership, cluster_names):
    """Count original entity labels in final cluster order, independently of aggregation."""
    rows = {cluster: i for i, cluster in enumerate(cluster_names)}
    counts = np.zeros((len(cluster_names), 2))
    for entity, label in labels.items():
        counts[rows[membership[entity]], label] += 1
    return counts


@pytest.mark.parametrize("axis", [0, 1])
def test_break_rebuilds_entity_and_interaction_classes_together(axis):
    from datasail.solver.overflow import break_cluster

    dataset, _ = isolated_dataset(axis)
    # Integer old IDs survive renumbering, hiding wrong rows behind conserved totals.
    old_ids = {old: i for i, old in enumerate(dataset.cluster_names)}
    dataset.cluster_names = [0, 1]
    dataset.cluster_map = {n: old_ids[c] for n, c in dataset.cluster_map.items()}
    dataset.cluster_weights = {old_ids[c]: w for c, w in dataset.cluster_weights.items()}
    dataset.cluster_interaction_stratification = {old_ids[c]: row for c, row in dataset.cluster_interaction_stratification.items()}
    labels = {n: int((i == 2) != bool(axis)) for i, n in enumerate(dataset.names)}
    dataset.classes = {0: 0, 1: 1}
    dataset.stratification = {n: np.eye(2)[label] for n, label in labels.items()}
    initial = entity_incidence_counts(labels, dataset.cluster_map, dataset.cluster_names)
    dataset.cluster_stratification = dict(zip(dataset.cluster_names, initial))
    break_cluster(dataset, 0, .4, "average")
    names = dataset.names
    assert dataset.cluster_map == {names[0]: 0, names[1]: 0, names[2]: 2, names[3]: 1}
    assert dataset.cluster_names == [0, 1, 2]
    expected = entity_incidence_counts(labels, dataset.cluster_map, dataset.cluster_names)
    literal = [[2, 0], [1, 0], [0, 1]] if axis == 0 else [[0, 2], [0, 1], [1, 0]]
    np.testing.assert_array_equal(expected, literal)
    actual = np.stack([dataset.cluster_stratification.get(c, np.zeros(2)) for c in dataset.cluster_names])
    np.testing.assert_array_equal(actual.sum(axis=0), initial.sum(axis=0))
    np.testing.assert_array_equal(actual, expected)
    interaction_expected = incidence_counts(INTERACTIONS, LABELS, dataset.cluster_map, dataset.cluster_names, axis)
    np.testing.assert_array_equal(np.stack([dataset.cluster_interaction_stratification[c] for c in dataset.cluster_names]), interaction_expected)
    np.testing.assert_array_equal(interaction_expected.sum(axis=0), [4, 4])


@pytest.mark.parametrize("axis,native", [(0, False), (1, False), (0, True), (1, True)])
def test_public_break_with_nonuniform_entity_and_interaction_classes(monkeypatch, axis, native):
    if native and "SCIP" not in cp.installed_solvers():
        pytest.skip("SCIP is unavailable")
    kwargs, interactions, interaction_labels = public_fixture()
    # First broken seven: 5 A/2 B; second: 2 A/5 B; survivors: 3 A/3 B.
    class_a = {0, 1, 2, 3, 4, 7, 8, 14, 15, 16}
    entity_labels = []
    for side, key in enumerate(("e_data", "f_data")):
        labels = {n: int((i not in class_a) != bool(side)) for i, n in enumerate(kwargs[key])}
        kwargs["e_strat" if side == 0 else "f_strat"] = {n: {label} for n, label in labels.items()}
        entity_labels.append(labels)
    prefix = "e" if axis == 0 else "f"
    similarities = np.eye(20)
    similarities[:14, :14] = 1
    kwargs.update({f"{prefix}_sim": (list(kwargs[f"{prefix}_data"]), similarities),
                   f"{prefix}_clusters": 7, "overflow": "break"})
    expected = {}
    roots = np.sqrt([8, 1, 1])
    targets = roots / roots.sum()
    inspected = []

    def observe_dispatch(**arguments):
        np.testing.assert_allclose(arguments["split_ratios"]["C2e"], targets, rtol=0, atol=1e-8)
        np.testing.assert_allclose(arguments["split_ratios"]["C2f"], targets, rtol=0, atol=1e-8)
        for side, key in enumerate(("e_dataset", "f_dataset")):
            dataset = arguments[key]
            entities = entity_incidence_counts(entity_labels[side], dataset.cluster_map, dataset.cluster_names)
            projected = incidence_counts(interactions, interaction_labels, dataset.cluster_map, dataset.cluster_names, side)
            assert len(np.unique(entities, axis=0)) > 1
            np.testing.assert_array_equal(entities.sum(axis=0), [10, 10])
            np.testing.assert_array_equal(projected.sum(axis=0), [20, 20])
            np.testing.assert_array_equal(np.stack([dataset.cluster_stratification.get(c, np.zeros(2)) for c in dataset.cluster_names]), entities)
            np.testing.assert_array_equal(np.stack([dataset.cluster_interaction_stratification[c] for c in dataset.cluster_names]), projected)
            if side == axis:
                assert len(dataset.cluster_names) == 8
                assert {tuple(row) for row in entities if row.sum() == 7} == {(5, 2), (2, 5)}
            expected[len(dataset.cluster_names)] = (entities, projected)
        return run_solver(**arguments)

    def inspect_model(loss, constraints, max_sec, solver, log_file):
        if native:
            from datasail.solver.utils import solve
            problem = solve(loss, constraints, max_sec, solver, log_file)
            assert problem is not None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            assert all(np.max(c.violation()) < 1e-7 for c in constraints)
            inspected.append(problem.status)
            return problem
        from tests.test_split_ratios import Model
        model = Model(constraints)
        for variable in model.variables:
            entities, projected = expected[variable.shape[1]]
            families = {tuple(counts.sum(axis=0)): (counts, limits) for counts, limits in model.class_bounds(variable)}
            assert len(families) == 2
            np.testing.assert_array_equal(families[(10, 10)][0], entities)
            np.testing.assert_array_equal(families[(20, 20)][0], projected)
            np.testing.assert_array_equal(families[(10, 10)][1], [[5, 5], [1, 1], [1, 1]])
            np.testing.assert_array_equal(families[(20, 20)][1], [[11, 11], [3, 3], [3, 3]])
            np.testing.assert_array_equal(model.lower_bounds(variable)[1], [11, 3, 3])
        inspected.append("constructed")
        return None

    monkeypatch.setattr("datasail.routine.run_solver", observe_dispatch)
    monkeypatch.setattr("datasail.solver.cluster_2d.solve", inspect_model)
    result = datasail(**kwargs)
    assert len(inspected) == 1
    if native:
        for side, output in enumerate(result[:2]):
            assignment = output["C2"][0]
            assert set(assignment) == set(entity_labels[side])
            names = kwargs["names"]
            counts = entity_incidence_counts(entity_labels[side], assignment, names)
            projected = incidence_counts(interactions, interaction_labels, assignment, names, side)
            np.testing.assert_array_less(np.array([[5, 5], [1, 1], [1, 1]]) - 1e-7, counts)
            np.testing.assert_array_less(np.array([[11, 11], [3, 3], [3, 3]]) - 1e-7, projected)
            assert all(sum(v == name for v in assignment.values()) >= minimum for name, minimum in zip(names, [11, 3, 3]))
