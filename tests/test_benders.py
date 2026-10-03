"""Benders 루프와 cut 유효성 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from cflp_bd_qa_ex2.benders.certificate import BoundState
from cflp_bd_qa_ex2.benders.cuts import build_optimality_cut, cut_is_valid_at
from cflp_bd_qa_ex2.benders.solver import GurobiMasterProvider, run_benders
from cflp_bd_qa_ex2.benders.trace import RECORD_TYPE_ITERATION, RECORD_TYPE_RUN
from cflp_bd_qa_ex2.models.cflp import solve_original_cflp
from cflp_bd_qa_ex2.models.subproblem import solve_subproblem
from cflp_bd_qa_ex2.status import Status


@pytest.mark.parametrize(
    "name,nf,nc,seed", [("4x12", 4, 12, 100), ("8x25", 8, 25, 200)]
)
def test_benders_matches_original(name, nf, nc, seed, pilot_config):
    """Benders + Gurobi 의 목적값이 Original CFLP 와 일치한다."""
    from cflp_bd_qa_ex2.data.generator import generate_instance

    inst = generate_instance(name, nf, nc, seed, pilot_config)
    ground_truth = solve_original_cflp(inst, pilot_config)
    trace = run_benders(inst, GurobiMasterProvider(), pilot_config)
    run = trace.run.fields

    assert ground_truth.status is Status.OPTIMAL
    assert run["termination_status"] == str(Status.OPTIMAL)
    assert run["best_upper_bound"] == pytest.approx(ground_truth.objective, rel=1e-6)


def test_lower_bound_is_monotone(small_instance, pilot_config):
    """certified lower bound 는 단조 증가한다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    bounds = [it.fields["certified_lower_bound"] for it in trace.iterations]
    assert all(b <= a + 1e-9 for b, a in zip(bounds, bounds[1:]))


def test_upper_bound_never_below_lower(small_instance, pilot_config):
    """UB < LB 가 발생하지 않는다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    for it in trace.iterations:
        assert it.fields["best_upper_bound"] >= it.fields["certified_lower_bound"] - 1e-6


def test_cut_is_valid_lower_bound(small_instance, pilot_config):
    """생성된 cut 은 임의의 y 에서 Q(y) 의 하한을 준다."""
    y_source = np.ones(small_instance.n_facilities, dtype=int)
    _, dual = solve_subproblem(small_instance, y_source, pilot_config)
    cut = build_optimality_cut(0, small_instance.capacity.astype(float), dual.u, dual.v, 1)

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(12):
        y = rng.integers(0, 2, small_instance.n_facilities)
        if float(small_instance.capacity @ y) < small_instance.total_demand:
            continue
        outcome, _ = solve_subproblem(small_instance, y, pilot_config)
        if outcome.status is not Status.OPTIMAL:
            continue
        assert cut_is_valid_at(cut, y, outcome.objective, tol=1e-6)
        checked += 1
    assert checked >= 1


def test_cut_is_tight_at_generating_y(small_instance, pilot_config):
    """cut 은 생성된 y 에서 Q(y) 와 같은 값을 준다 (strong duality)."""
    y = np.ones(small_instance.n_facilities, dtype=int)
    outcome, dual = solve_subproblem(small_instance, y, pilot_config)
    cut = build_optimality_cut(0, small_instance.capacity.astype(float), dual.u, dual.v, 1)
    assert cut.lower_bound_at(y) == pytest.approx(outcome.objective, abs=1e-6)


def test_cut_direction(small_instance, pilot_config):
    """residual 과 violation 의 부호 규약이 일관된다."""
    y = np.ones(small_instance.n_facilities, dtype=int)
    outcome, dual = solve_subproblem(small_instance, y, pilot_config)
    cut = build_optimality_cut(0, small_instance.capacity.astype(float), dual.u, dual.v, 1)
    bound = cut.lower_bound_at(y)
    assert cut.violation(y, bound + 1.0) == 0.0
    assert cut.violation(y, bound - 1.0) == pytest.approx(1.0)


def test_record_types_are_separated(small_instance, pilot_config):
    """MP-level 과 Benders-level 기록이 다른 record_type 을 갖는다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    records = trace.to_records()
    types = {r["record_type"] for r in records}
    assert types == {RECORD_TYPE_ITERATION, RECORD_TYPE_RUN}


def test_bound_state_gap():
    """relative gap 계산이 정의대로 동작한다."""
    state = BoundState()
    state.update_lower(90.0, "test", Status.OPTIMAL)
    state.update_upper(100.0, [1])
    assert state.relative_gap() == pytest.approx(0.1)
    assert state.check_bound_order(1e-9) is None


def test_bound_state_detects_inversion():
    """UB < LB 를 검출한다."""
    state = BoundState()
    state.update_lower(100.0, "test", Status.OPTIMAL)
    state.update_upper(90.0, [1])
    assert state.check_bound_order(1e-9) is not None


def test_duplicate_cut_status_is_separate_field(small_instance, pilot_config):
    """duplicate-cut 진단이 termination_status 와 별개 열로 저장된다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    record = trace.run.as_record()
    assert "duplicate_cut_status" in record
    assert "num_duplicate_cuts" in record
    assert "first_duplicate_iteration" in record
    assert record["duplicate_cut_status"] != record["termination_status"]


def test_duplicate_alone_does_not_terminate(small_instance, pilot_config):
    """duplicate 만으로는 종료하지 않는다 (gap 미개선이 함께 필요)."""
    from cflp_bd_qa_ex2.status import DuplicateCutStatus

    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    for iteration in trace.iterations:
        fields = iteration.as_record()
        if fields.get("duplicate_cut") and fields.get("gap_improved"):
            # 이 iteration 에서 종료되지 않았어야 한다.
            assert iteration is not trace.iterations[-1] or (
                trace.run.fields["termination_status"] != str(Status.STALLED_DUPLICATE_CUT)
            )
    assert trace.run.fields["duplicate_cut_status"] in (
        str(DuplicateCutStatus.NONE),
        str(DuplicateCutStatus.DUPLICATE_DETECTED),
    )
