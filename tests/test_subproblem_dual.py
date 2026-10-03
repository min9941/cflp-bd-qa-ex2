"""Subproblem strong duality 와 dual minimality guard 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from cflp_bd_qa_ex2.models.subproblem import (
    check_dual_minimality,
    compute_minimal_v,
    dual_continuation_allowed,
    solve_subproblem,
)
from cflp_bd_qa_ex2.status import DualMinimalityStatus, Status


def _all_open(inst):
    """모든 시설을 여는 y."""
    return np.ones(inst.n_facilities, dtype=int)


def test_strong_duality(small_instance, pilot_config):
    """primal 목적값과 dual 목적값이 일치한다."""
    outcome, dual = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    assert outcome.status is Status.OPTIMAL
    assert dual.strong_duality_gap < 1e-6


def test_dual_is_feasible(small_instance, pilot_config):
    """dual constraint u_i - d_i v_j <= d_i c_ij 를 만족한다."""
    _, dual = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    assert dual.dual_constraint_max_violation < 1e-6
    assert (dual.v >= -1e-12).all()


def test_sp_feasible_when_capacity_suffices(small_instance, pilot_config):
    """aggregate capacity 가 충분하면 SP 는 feasible 하다."""
    outcome, _ = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    assert outcome.status is Status.OPTIMAL


def test_sp_infeasible_when_all_closed(small_instance, pilot_config):
    """모든 시설이 닫히면 SP 는 infeasible 하며 임의 cut 을 만들지 않는다."""
    y = np.zeros(small_instance.n_facilities, dtype=int)
    outcome, dual = solve_subproblem(small_instance, y, pilot_config)
    assert outcome.status is not Status.OPTIMAL
    assert dual is None


def test_minimal_v_is_feasible(small_instance, pilot_config):
    """투영된 v 도 dual feasible 하다."""
    _, dual = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    v_min = compute_minimal_v(dual.u, small_instance.demand, small_instance.transport_cost)
    lhs = dual.u[:, None] - small_instance.demand[:, None].astype(float) * v_min[None, :]
    rhs = small_instance.demand[:, None].astype(float) * small_instance.transport_cost
    assert float(np.maximum(0.0, lhs - rhs).max()) < 1e-9


def test_gurobi_dual_is_componentwise_minimal(small_instance, pilot_config):
    """Gurobi 가 반환하는 dual 이 componentwise 최소 v 와 일치한다.

    이는 수학적 필연이 아니라 관측된 solver 특성이다. guard 는 이것이
    깨지는 경우를 탐지하기 위한 것이다.
    """
    y = _all_open(small_instance)
    _, dual = solve_subproblem(small_instance, y, pilot_config)
    dual = check_dual_minimality(dual, y, 1e-7, 1e-7)
    assert dual.minimality_status is DualMinimalityStatus.MINIMAL


def test_guard_detects_inflated_closed_facility_v(small_instance, pilot_config):
    """닫힌 시설의 v_j 를 인위로 부풀리면 guard 가 검출한다."""
    from cflp_bd_qa_ex2.models.cflp import solve_original_cflp

    y = np.asarray(solve_original_cflp(small_instance, pilot_config).payload["y"], dtype=int)
    closed = int(np.where(y == 0)[0][0])
    outcome, dual = solve_subproblem(small_instance, y, pilot_config)
    assert outcome.status is Status.OPTIMAL

    dual.v = dual.v.copy()
    dual.v[closed] += 1000.0  # 닫힌 시설의 v 팽창
    dual = check_dual_minimality(dual, y, 1e-7, 1e-7)
    assert dual.minimality_status is DualMinimalityStatus.DUAL_NOT_MINIMAL
    assert closed in dual.violating_facilities
    assert dual.v_minimality_closed_gap > 100.0
    assert dual.v_minimality_open_gap < 1e-6


def test_inflated_closed_v_keeps_cut_valid_but_grows_slack(small_instance, pilot_config):
    """팽창된 v 로 만든 cut 도 valid 하지만 cut slack 범위가 커진다.

    즉 깨지는 것은 정확성이 아니라 bit 수의 재현성이다.
    """
    from cflp_bd_qa_ex2.benders.cuts import build_optimality_cut, cut_is_valid_at
    from cflp_bd_qa_ex2.encoding.slack import cut_slack_max
    from cflp_bd_qa_ex2.models.cflp import solve_original_cflp

    y = np.asarray(solve_original_cflp(small_instance, pilot_config).payload["y"], dtype=int)
    closed = int(np.where(y == 0)[0][0])
    outcome, dual = solve_subproblem(small_instance, y, pilot_config)

    inflated = dual.v.copy()
    inflated[closed] += 1000.0
    capacity = small_instance.capacity.astype(float)

    raw_cut = build_optimality_cut(0, capacity, dual.u, dual.v, 1)
    inflated_cut = build_optimality_cut(0, capacity, dual.u, inflated, 1)

    assert cut_is_valid_at(raw_cut, y, outcome.objective, tol=1e-6)
    assert cut_is_valid_at(inflated_cut, y, outcome.objective, tol=1e-6)

    theta_upper = small_instance.objective_upper_bound
    raw_max = cut_slack_max(theta_upper, capacity, dual.v, dual.u.sum())
    inflated_max = cut_slack_max(theta_upper, capacity, inflated, dual.u.sum())
    assert inflated_max > raw_max


def test_continuation_requires_sp_optimality(small_instance, pilot_config):
    """SP 가 OPTIMAL 이 아니면 continue 가 허용되지 않는다."""
    y = _all_open(small_instance)
    _, dual = solve_subproblem(small_instance, y, pilot_config)
    allowed, _ = dual_continuation_allowed(Status.INFEASIBLE, dual, 1e-9, 1e-6)
    assert allowed is False
    allowed, _ = dual_continuation_allowed(Status.OPTIMAL, dual, 1e-9, 1e-6)
    assert allowed is True


def test_continuation_rejects_dual_infeasibility(small_instance, pilot_config):
    """dual constraint 가 위반되면 continue 가 허용되지 않는다."""
    _, dual = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    dual.dual_constraint_max_violation = 1.0
    allowed, reason = dual_continuation_allowed(Status.OPTIMAL, dual, 1e-9, 1e-6)
    assert allowed is False
    assert "dual constraint" in reason


def test_dual_vector_hash_is_stable(small_instance, pilot_config):
    """같은 dual 은 같은 해시를 준다."""
    _, a = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    _, b = solve_subproblem(small_instance, _all_open(small_instance), pilot_config)
    assert a.dual_vector_hash == b.dual_vector_hash
