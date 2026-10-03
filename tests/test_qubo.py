"""QUBO builder 테스트: energy 일치, 계수 범위 정의, role table."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cflp_bd_qa_ex2.benders.cuts import build_optimality_cut
from cflp_bd_qa_ex2.models.subproblem import solve_subproblem
from cflp_bd_qa_ex2.qubo.builder import (
    ROLE_CAPACITY_SLACK,
    ROLE_CUT_SLACK,
    ROLE_THETA,
    ROLE_Y,
    AffineForm,
    build_master_qubo,
)
from cflp_bd_qa_ex2.qubo.diagnostics import (
    logical_graph,
    qubo_metrics,
    role_counts,
    source_graph_fingerprint,
    variable_role_table,
)


def _one_cut(inst, cfg):
    """모든 시설을 연 상태에서 cut 하나를 만든다."""
    y = np.ones(inst.n_facilities, dtype=int)
    _, dual = solve_subproblem(inst, y, cfg)
    return build_optimality_cut(0, inst.capacity.astype(float), dual.u, dual.v, 1)


def test_energy_matches_bqm(small_instance, test_config):
    """직접 계산한 energy 와 dimod BQM energy 가 일치한다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    bqm = qubo.to_bqm()
    rng = random.Random(0)
    for _ in range(30):
        assignment = {v: rng.randint(0, 1) for v in qubo.variable_labels}
        assert qubo.energy(assignment) == pytest.approx(bqm.energy(assignment), rel=1e-9)


def test_energy_matches_bqm_with_cuts(small_instance, test_config):
    """cut 이 있는 QUBO 에서도 energy 가 일치한다."""
    cut = _one_cut(small_instance, test_config)
    qubo = build_master_qubo(small_instance, [cut], test_config)
    bqm = qubo.to_bqm()
    rng = random.Random(1)
    for _ in range(30):
        assignment = {v: rng.randint(0, 1) for v in qubo.variable_labels}
        assert qubo.energy(assignment) == pytest.approx(bqm.energy(assignment), rel=1e-9)


def test_feasible_assignment_has_objective_energy(small_instance, test_config):
    """residual 이 0인 배정의 energy 는 목적값과 같다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    assignment = {v: 0 for v in qubo.variable_labels}
    for name in qubo.variables_by_role(ROLE_Y):
        assignment[name] = 1
    slack = small_instance.total_capacity - small_instance.total_demand
    bits = qubo.capacity_slack.encoding.encode_nearest(float(slack))
    for name, bit in zip(qubo.capacity_slack.encoding.variable_names(), bits):
        assignment[name] = bit

    residual = qubo.residual_forms["capacity"].evaluate(assignment)
    assert residual == pytest.approx(0.0, abs=1e-9)
    assert qubo.energy(assignment) == pytest.approx(float(small_instance.fixed_cost.sum()), rel=1e-9)


def test_variable_roles_and_counts(small_instance, test_config):
    """변수 role 구성이 예상과 일치한다."""
    cut = _one_cut(small_instance, test_config)
    qubo = build_master_qubo(small_instance, [cut], test_config)
    counts = role_counts(qubo)
    assert counts["num_y_variables"] == small_instance.n_facilities
    assert counts["num_theta_bits"] == int(test_config["encoding"]["theta"]["bits"])
    assert counts["num_capacity_slack_bits"] == qubo.capacity_slack.capacity_slack_bits
    assert counts["num_cut_slack_bits"] == qubo.cut_slacks[0].cut_slack_bits
    assert sum(counts.values()) == qubo.num_variables
    assert set(qubo.variable_roles.values()) <= {
        ROLE_Y, ROLE_THETA, ROLE_CAPACITY_SLACK, ROLE_CUT_SLACK
    }


def test_coefficient_range_is_ratio(small_instance, test_config):
    """coefficient range 는 차이가 아니라 비율이다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    metrics = qubo_metrics(qubo)
    assert metrics["qubo_coefficient_range"] == pytest.approx(
        metrics["qubo_max_abs_coefficient"] / metrics["qubo_min_abs_coefficient"]
    )
    assert metrics["qubo_coefficient_range"] >= 1.0


def test_density_definition(small_instance, test_config):
    """density = |E| / (n(n-1)/2)."""
    qubo = build_master_qubo(small_instance, [], test_config)
    metrics = qubo_metrics(qubo)
    n = metrics["qubo_num_variables"]
    assert metrics["qubo_density"] == pytest.approx(
        metrics["qubo_num_quadratic_terms"] / (n * (n - 1) / 2)
    )


def test_qubo_grows_with_cuts(small_instance, test_config):
    """cut 을 추가하면 변수와 엣지가 늘어난다."""
    cut = _one_cut(small_instance, test_config)
    before = build_master_qubo(small_instance, [], test_config)
    after = build_master_qubo(small_instance, [cut], test_config)
    assert after.num_variables > before.num_variables
    assert len(after.quadratic) > len(before.quadratic)


def test_source_graph_includes_all_variables(small_instance, test_config):
    """source graph 는 isolated logical variable 도 포함한다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    graph = logical_graph(qubo)
    assert graph.number_of_nodes() == qubo.num_variables
    assert set(graph.nodes()) == set(qubo.variable_labels)


def test_source_graph_fingerprint_is_stable(small_instance, test_config):
    """같은 QUBO 는 같은 지문을 준다."""
    a = source_graph_fingerprint(logical_graph(build_master_qubo(small_instance, [], test_config)))
    b = source_graph_fingerprint(logical_graph(build_master_qubo(small_instance, [], test_config)))
    assert a == b


def test_role_table_covers_all_variables(small_instance, test_config):
    """role table 이 모든 변수를 포함한다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    table = variable_role_table(qubo)
    assert len(table) == qubo.num_variables
    assert {row["variable"] for row in table} == set(qubo.variable_labels)


def test_affine_square_expansion():
    """AffineForm.square_into 가 (a+b+c)^2 를 정확히 전개한다."""
    form = AffineForm().add("a", 2.0).add("b", -3.0).add_constant(1.5)
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    offset = form.square_into(linear, quadratic, 1.0)

    for a_val in (0, 1):
        for b_val in (0, 1):
            assignment = {"a": a_val, "b": b_val}
            expected = form.evaluate(assignment) ** 2
            actual = offset + sum(c * assignment[v] for v, c in linear.items())
            actual += sum(c * assignment[x] * assignment[y] for (x, y), c in quadratic.items())
            assert actual == pytest.approx(expected)
