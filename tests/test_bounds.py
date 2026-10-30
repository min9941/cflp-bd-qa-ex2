"""Theta bound 전략과 offline 분석 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from cflp_bd_qa_ex2.encoding.bounds import (
    STRATEGY_IDS,
    BoundError,
    build_all_bounds,
    build_bound,
    lower_bound_assignment_relaxation,
    upper_bound_baseline,
    upper_bound_cost_based,
    validate_bound,
)
from cflp_bd_qa_ex2.encoding.theta import bits_for_target_resolution, build_theta_encoding
from cflp_bd_qa_ex2.evaluation.bound_analysis import CutRecord, analyze_configuration


def test_upper_bounds_are_ordered(small_instance):
    """S1 의 상한은 S0 보다 작거나 같다."""
    assert upper_bound_cost_based(small_instance) <= upper_bound_baseline(small_instance)


def test_lower_bound_is_below_upper(small_instance):
    """assignment relaxation 하한이 상한보다 작다."""
    assert lower_bound_assignment_relaxation(small_instance) < upper_bound_cost_based(
        small_instance
    )


def test_all_bounds_are_valid_at_optimum(small_instance, pilot_config):
    """최적해의 Q(y*) 가 모든 전략의 구간 안에 있다."""
    from cflp_bd_qa_ex2.models.cflp import solve_original_cflp
    from cflp_bd_qa_ex2.models.subproblem import solve_subproblem
    from cflp_bd_qa_ex2.status import Status

    ground_truth = solve_original_cflp(small_instance, pilot_config)
    assert ground_truth.status is Status.OPTIMAL
    y_star = np.asarray(ground_truth.payload["y"], dtype=int)
    outcome, _ = solve_subproblem(small_instance, y_star, pilot_config)
    recourse = float(outcome.objective)

    bounds = build_all_bounds(small_instance, pilot_config)
    for strategy_id in STRATEGY_IDS:
        report = validate_bound(bounds[strategy_id], recourse)
        assert report["bound_valid"], f"{strategy_id} 가 Q(y*) 를 벗어난다: {report}"
        assert report["theta_clipping_count"] == 0


def test_s3_lower_bound_is_at_least_s2(small_instance, pilot_config):
    """Q(1) 은 assignment relaxation 하한보다 작지 않다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    assert bounds["S3"].theta_lower_bound >= bounds["S2"].theta_lower_bound - 1e-9
    # 상한은 S1~S3 가 같다
    assert bounds["S1"].theta_upper_bound == pytest.approx(bounds["S2"].theta_upper_bound)
    assert bounds["S2"].theta_upper_bound == pytest.approx(bounds["S3"].theta_upper_bound)


def test_unknown_strategy_raises(small_instance, pilot_config):
    """알 수 없는 전략은 거부한다."""
    with pytest.raises(BoundError, match="알 수 없는"):
        build_bound("S9", small_instance, pilot_config)


def test_theta_encoding_accepts_lower_bound(small_instance):
    """theta_L 을 주면 decode 끝점이 그 값에서 시작한다."""
    encoding = build_theta_encoding(100.0, 4, theta_lower_bound=40.0)
    assert encoding.decode([0] * 4) == pytest.approx(40.0)
    assert encoding.decode([1] * 4) == pytest.approx(100.0)
    assert encoding.theta_delta == pytest.approx(60.0 / 15)
    # 기본값은 기존 동작을 유지한다
    assert build_theta_encoding(100.0, 4).theta_lower_bound == 0.0


def test_target_resolution_bits_are_upper_bounded(small_instance):
    """실제 해상도는 목표값과 같지 않고 **이하**이다."""
    target = 10.0
    bits = bits_for_target_resolution(0.0, 100.0, target)
    actual = 100.0 / (2**bits - 1)
    assert actual <= target + 1e-12
    # ceil 때문에 보통 같지 않다
    assert bits == 4 and actual == pytest.approx(100.0 / 15)

    with pytest.raises(Exception):
        bits_for_target_resolution(0.0, 100.0, 0.0)


def _cut(u_sum: float, v: np.ndarray, iteration: int = 1) -> CutRecord:
    """합이 u_sum 인 더미 dual 로 CutRecord 를 만든다."""
    return CutRecord(
        iteration=iteration,
        dual_u=np.full(1, u_sum),
        dual_v=v,
        activated=True,
    )


def test_raising_lower_bound_never_reduces_cut_slack_bits(small_instance, pilot_config):
    """정리: theta_U 고정 + theta_L 증가 -> cut slack bit 는 감소하지 않는다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    rng = np.random.default_rng(0)
    cuts = [
        _cut(float(rng.uniform(0, 500)), rng.uniform(0, 2, small_instance.n_facilities), i)
        for i in range(1, 11)
    ]

    totals = {}
    for strategy_id in ("S1", "S2", "S3"):
        result = analyze_configuration(
            small_instance, bounds[strategy_id], cuts, theta_bits=10
        )
        totals[strategy_id] = result["cut_slack_bits_total"]

    assert totals["S1"] <= totals["S2"] <= totals["S3"], totals


def test_analyze_requires_exactly_one_bit_mode(small_instance, pilot_config):
    """theta_bits 와 target_delta 중 정확히 하나만 받는다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    with pytest.raises(ValueError):
        analyze_configuration(small_instance, bounds["S1"], [], theta_bits=10, target_delta=1.0)
    with pytest.raises(ValueError):
        analyze_configuration(small_instance, bounds["S1"], [])


def test_analyze_reports_negative_a_cuts(small_instance, pilot_config):
    """A_k < 0 인 cut 을 세어 보고한다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    capacity = small_instance.capacity.astype(float)
    # v 를 크게 주면 A_k = sum(u) - sum(s*v) 가 음수가 된다
    negative = _cut(0.0, np.full(small_instance.n_facilities, 10.0))
    positive = _cut(1000.0, np.zeros(small_instance.n_facilities))

    result = analyze_configuration(
        small_instance, bounds["S1"], [negative, positive], theta_bits=10
    )
    assert result["num_cuts_with_negative_a"] == 1
    assert result["a_min"] < 0 < result["a_max"]
    assert float(capacity @ negative.dual_v) > 0
