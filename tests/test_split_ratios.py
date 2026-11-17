"""Two-dimensional constraints consume the final preprocessed per-axis targets."""
import math

import cvxpy as cp
import numpy as np
import pytest

from datasail.sail import datasail
from datasail.solver.cluster_2d import solve_c2


def axis_targets(requested):
    roots = np.sqrt(requested)
    return roots / roots.sum()


class Model:
    """Inspect affine constraint constants and coefficients without stack inspection."""

    def __init__(self, constraints):
        self.constraints = constraints
        self.variables = list({v.id: v for c in constraints for v in c.variables()}.values())
        for variable in self.variables:
            variable.value = np.zeros(variable.shape)

    def lower_bounds(self, variable):
        limits = np.full(variable.shape[0], np.nan)
        weights = None
        for constraint in self.constraints:
            if not isinstance(constraint, cp.constraints.nonpos.Inequality) or constraint.shape != ():
                continue
            left, right = constraint.args
            if not left.is_constant() or [v.id for v in right.variables()] != [variable.id]:
                continue
            coefficients = right.grad[variable].toarray().reshape(variable.shape, order="F")
            rows = np.flatnonzero(np.any(coefficients != 0, axis=1))
            assert len(rows) == 1
            limits[rows[0]] = float(left.value)
            weights = coefficients[rows[0]]
        assert np.isfinite(limits).all(), "An axis is missing a weight constraint"
        return weights, limits

    def class_bounds(self, variable):
        result = []
        for constraint in self.constraints:
            if not isinstance(constraint, cp.constraints.nonpos.Inequality) or len(constraint.shape) != 2:
                continue
            left, right = constraint.args
            if not left.is_constant() or [v.id for v in right.variables()] != [variable.id]:
                continue
            gradient = right.grad[variable].toarray()
            counts = np.column_stack([
                gradient[:, k * variable.shape[0]].reshape(variable.shape, order="F")[0]
                for k in range(constraint.shape[1])
            ])
            result.append((counts, left.value))
        return result


@pytest.fixture
def models(monkeypatch):
    captured = []
    for technique, module in [("C2", "cluster_2d"), ("I2", "id_2d"), ("C1e", "cluster_1d"), ("I1e", "id_1d")]:
        def capture(loss, constraints, max_sec, solver, log_file, technique=technique):
            captured.append((technique, Model(constraints)))
            return None
        monkeypatch.setattr(f"datasail.solver.{module}.solve", capture)
    return captured


def public_input(techniques, requested, n=20, strata=False, overflow="assign"):
    e = [f"e{i}" for i in range(n)]
    f = [f"f{i}" for i in range(n)]
    kwargs = dict(
        techniques=techniques, splits=requested, names=[f"s{i}" for i in range(len(requested))],
        e_type="O", e_data={x: str(i) for i, x in enumerate(e)}, e_sim=(e, np.eye(n)),
        f_type="O", f_data={x: str(i) for i, x in enumerate(f)}, f_sim=(f, np.eye(n)),
        e_weights={x: 100 / n for x in e}, f_weights={x: 100 / n for x in f},
        e_clusters=n, f_clusters=n, inter=list(zip(e, f)),
        epsilon=.05, delta=.05, max_sec=2, threads=1, overflow=overflow,
    )
    if strata:
        kwargs["e_strat"] = {x: {int(i >= n // 2)} for i, x in enumerate(e)}
        kwargs["f_strat"] = {x: {int(i >= n // 2)} for i, x in enumerate(f)}
    return kwargs


@pytest.mark.parametrize("technique", ["C2", "I2"])
@pytest.mark.parametrize("requested, expected", [([8, 1, 1], [55, 19, 19]), ([8, 2], [63, 31])])
def test_public_unequal_ratio_weight_limits(models, technique, requested, expected):
    datasail(**public_input([technique], requested))
    assert len(models) == 1
    independently_calculated = np.maximum(np.floor(axis_targets(requested) * .95 * 100), 1)
    np.testing.assert_array_equal(independently_calculated, expected)
    for variable in models[0][1].variables:
        weights, limits = models[0][1].lower_bounds(variable)
        assert weights.sum() == 100
        np.testing.assert_array_equal(limits, expected)


@pytest.mark.parametrize("technique", ["C2", "I2"])
def test_entity_classes_and_weights_admit_a_common_assignment(models, technique):
    datasail(**public_input([technique], [8, 1, 1], n=100, strata=True))
    model = models[0][1]
    assignment = np.zeros((3, 100))
    for offset in (0, 50):
        start = offset
        for split, count in enumerate([28, 11, 11]):
            assignment[split, start:start + count] = 1
            start += count
    for variable in model.variables:
        weights, limits = model.lower_bounds(variable)
        [(counts, class_limits)] = model.class_bounds(variable)
        np.testing.assert_array_equal(weights, np.ones(100))
        np.testing.assert_array_equal(counts.sum(axis=0), [50, 50])
        combined = np.maximum(limits, class_limits.sum(axis=1))
        assert combined.sum() <= 100, {"overall": limits, "classes": class_limits, "combined": combined}
        np.testing.assert_array_equal(limits, [55, 19, 19])
        np.testing.assert_array_equal(class_limits, [[27, 27], [9, 9], [9, 9]])
        variable.value = assignment
    np.testing.assert_array_equal(assignment.sum(axis=1), [56, 22, 22])
    for constraint in model.constraints:
        assert np.max(constraint.violation()) == 0


def interaction_input(technique):
    kwargs = public_input([technique], [8, 1, 1], n=10)
    f = [f"f{i}" for i in range(12)]
    kwargs.update(f_data={x: str(i) for i, x in enumerate(f)}, f_sim=(f, np.eye(12)),
                  f_weights={x: 100 / 12 for x in f}, f_clusters=12)
    interactions = [(e, other) for e in kwargs["e_data"] for other in f]
    labels = {pair: {int((i + 2 * j) % 3 == 0)}
              for i, e in enumerate(kwargs["e_data"]) for j, other in enumerate(f)
              for pair in [(e, other)]}
    kwargs.update(inter=interactions, inter_strat=labels)
    # Count each explicitly enumerated interaction once on each axis, independently of the reader.
    projected = []
    for axis, names in enumerate((kwargs["e_data"], kwargs["f_data"])):
        counts = {name: [0, 0] for name in names}
        for pair, label in labels.items():
            counts[pair[axis]][next(iter(label))] += 1
        projected.append(np.array(list(counts.values())))
    return kwargs, projected


@pytest.mark.parametrize("technique", ["C2", "I2"])
def test_public_projected_interaction_classes(models, technique):
    kwargs, projected = interaction_input(technique)
    datasail(**kwargs)
    model = models[0][1]
    for variable in model.variables:
        expected_counts = next(m for m in projected if len(m) == variable.shape[1])
        [(counts, limits)] = model.class_bounds(variable)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(counts.sum(axis=0), [80, 40])
        expected = np.maximum(np.floor(axis_targets([8, 1, 1])[:, None] * .95 * expected_counts.sum(axis=0)), 1)
        np.testing.assert_array_equal(limits, expected)


def test_direct_c2_projected_cluster_counts(models):
    _, projected = interaction_input("C2")
    # Explicit cluster {e0,e1}; all other e/f entities remain singletons.
    e_counts = np.vstack([projected[0][:2].sum(axis=0), projected[0][2:]])
    targets = axis_targets([8, 1, 1]).tolist()
    solve_c2(
        e_clusters=list(range(9)), e_s_matrix=None, eis_matrix=e_counts,
        e_similarities=np.eye(9), e_distances=None, e_weights=np.array([2] + [1] * 8),
        e_splits=targets, e_names=["s0", "s1", "s2"],
        f_clusters=list(range(12)), f_s_matrix=None, fis_matrix=projected[1],
        f_similarities=np.eye(12), f_distances=None, f_weights=np.ones(12),
        f_splits=targets, f_names=["s0", "s1", "s2"],
        epsilon=.05, delta=.05, max_sec=2, solver="SCIP", log_file=None,
    )
    model = models[0][1]
    for variable in model.variables:
        expected_counts = e_counts if variable.shape[1] == 9 else projected[1]
        [(counts, limits)] = model.class_bounds(variable)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(limits, np.maximum(np.floor(np.array(targets)[:, None] * .95 * [80, 40]), 1))


@pytest.mark.parametrize("technique", ["C2", "I2"])
@pytest.mark.parametrize("strategy", ["assign", "break"])
def test_no_overflow_modes(models, technique, strategy):
    datasail(**public_input([technique], [8, 1, 1], overflow=strategy))
    for variable in models[0][1].variables:
        np.testing.assert_array_equal(models[0][1].lower_bounds(variable)[1], [55, 19, 19])


@pytest.mark.parametrize("strategy", ["assign", "break"])
def test_real_cluster_overflow_targets_are_not_reconverted(models, strategy):
    kwargs = public_input(["C2"], [8, 2, 1], overflow=strategy)
    names = list(kwargs["e_data"])
    similarities = np.eye(20)
    similarities[:14, :14] = 1
    kwargs.update(e_sim=(names, similarities), e_clusters=7)
    datasail(**kwargs)
    model = models[0][1]
    variable = next(v for v in model.variables if v.shape[1] != 20)
    weights, limits = model.lower_bounds(variable)
    if strategy == "assign":
        assert variable.shape == (2, 6)
        assert weights.sum() == 30
        # Removing the first split renormalizes the remaining sqrt(2):sqrt(1) targets.
        expected = np.floor(axis_targets([2, 1]) * .95 * 30)
        np.testing.assert_array_equal(limits, [16, 11])
    else:
        assert variable.shape == (3, 8)
        assert weights.sum() == 100
        expected = np.floor(axis_targets([8, 2, 1]) * .95 * 100)
    np.testing.assert_array_equal(limits, expected)


@pytest.mark.parametrize("strategy", ["assign", "break"])
def test_identity_overflow_preserves_remaining_targets(models, strategy):
    kwargs = public_input(["I2"], [8, 2, 1], overflow=strategy)
    kwargs["e_weights"] = {name: 70 if i == 0 else 30 / 19 for i, name in enumerate(kwargs["e_data"])}
    datasail(**kwargs)
    model = models[0][1]
    variable = next(v for v in model.variables if v.shape[1] == 19)
    weights, limits = model.lower_bounds(variable)
    assert math.isclose(weights.sum(), 30)
    np.testing.assert_array_equal(limits, [16, 11])


@pytest.mark.parametrize("technique", ["C2", "I2"])
def test_equal_ratio_control(models, technique):
    datasail(**public_input([technique], [1, 1]))
    for variable in models[0][1].variables:
        np.testing.assert_array_equal(models[0][1].lower_bounds(variable)[1], [47, 47])


@pytest.mark.parametrize("technique", ["C1e", "I1e"])
def test_one_dimensional_targets_are_unchanged(models, technique):
    datasail(**public_input([technique], [8, 2]))
    [variable] = models[0][1].variables
    np.testing.assert_array_equal(models[0][1].lower_bounds(variable)[1], [76, 19])


@pytest.mark.parametrize("techniques", [["I2", "C2"], ["C2", "I2"]])
def test_combined_techniques_without_overflow(models, techniques):
    datasail(**public_input(techniques, [8, 1, 1]))
    assert [tech for tech, _ in models] == techniques
    for _, model in models:
        for variable in model.variables:
            np.testing.assert_array_equal(model.lower_bounds(variable)[1], [55, 19, 19])


@pytest.mark.skipif("SCIP" not in cp.installed_solvers(), reason="SCIP is unavailable")
@pytest.mark.parametrize("technique, module", [("C2", "cluster_2d"), ("I2", "id_2d")])
def test_tiny_native_scip_feasibility(monkeypatch, technique, module):
    from datasail.solver.utils import solve

    checked = []

    def native(loss, constraints, max_sec, solver, log_file):
        problem = solve(loss, constraints, max_sec, solver, log_file)
        assert problem is not None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert all(np.max(c.violation()) < 1e-7 for c in constraints)
        checked.append(problem.status)
        return problem

    monkeypatch.setattr(f"datasail.solver.{module}.solve", native)
    kwargs = public_input([technique], [8, 2], strata=True)
    e, f, _ = datasail(**kwargs)
    for result in (e, f):
        assignment = result[technique][0]
        assert len(assignment) == 20
        totals = [sum(split == f"s{i}" for split in assignment.values()) for i in range(2)]
        assert totals[0] * 5 >= 63 and totals[1] * 5 >= 31
    assert len(checked) == 1
