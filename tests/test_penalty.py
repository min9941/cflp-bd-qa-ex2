"""Penalty 계수와 ground state 검증 테스트."""

from __future__ import annotations

import pytest

from cflp_bd_qa_ex2.qubo.builder import build_master_qubo
from cflp_bd_qa_ex2.qubo.exact import (
    exhaustive_ground_state,
    miqp_ground_state,
    validate_qubo_exactly,
)
from cflp_bd_qa_ex2.qubo.penalty import build_penalty_spec
from cflp_bd_qa_ex2.status import Status


def test_penalty_alpha_rule(small_instance, pilot_config):
    """penalty_alpha = margin * U_obj 이며 모든 constraint 에 동일하다."""
    spec = build_penalty_spec(small_instance.objective_upper_bound, 3, pilot_config)
    expected = 1.1 * small_instance.objective_upper_bound
    assert spec.penalty_alpha_capacity == pytest.approx(expected)
    assert len(spec.penalty_alpha_cuts) == 3
    assert all(value == pytest.approx(expected) for value in spec.penalty_alpha_cuts)


def test_penalty_name_is_penalty_alpha(small_instance, pilot_config):
    """결과 field 이름이 penalty_alpha 계열이다 (단독 alpha 금지)."""
    spec = build_penalty_spec(small_instance.objective_upper_bound, 1, pilot_config)
    record = spec.as_record()
    assert "penalty_alpha_capacity" in record
    assert "alpha" not in record


def test_exhaustive_skips_when_too_large(small_instance, pilot_config):
    """변수 수가 threshold 를 넘으면 임의 축소 없이 건너뛴다."""
    qubo = build_master_qubo(small_instance, [], pilot_config)
    result = exhaustive_ground_state(qubo, max_variables=5)
    assert result.status is Status.EXACT_QUBO_SKIPPED_TOO_LARGE
    assert result.ground_state is None


def test_exhaustive_ground_state_is_feasible(tiny_instance, test_config):
    """작은 QUBO 의 ground state 는 MP-feasible 하다 (penalty 충분)."""
    qubo = build_master_qubo(tiny_instance, [], test_config)
    result = validate_qubo_exactly(qubo, tiny_instance, [], test_config)
    assert result.status is Status.EXACT_QUBO_VALIDATED
    assert result.penalty_status is Status.SUCCESS
    assert result.fields["ground_state_mp_feasible"] is True


def test_miqp_matches_exhaustive(tiny_instance, test_config):
    """Gurobi MIQP ground state 가 exhaustive 결과와 일치한다."""
    qubo = build_master_qubo(tiny_instance, [], test_config)
    brute = exhaustive_ground_state(qubo, max_variables=30)
    miqp = miqp_ground_state(qubo, test_config)
    if miqp.status is Status.EXACT_QUBO_VALIDATED:
        assert miqp.ground_state_energy == pytest.approx(brute.ground_state_energy, rel=1e-6)


def test_miqp_respects_size_limit(small_instance, pilot_config):
    """MIQP 변수 상한을 넘으면 EXACT_QUBO_SKIPPED_TOO_LARGE."""
    import copy

    cfg = copy.deepcopy(pilot_config)
    cfg["validation"]["miqp_max_variables"] = 5
    qubo = build_master_qubo(small_instance, [], cfg)
    result = miqp_ground_state(qubo, cfg)
    assert result.status is Status.EXACT_QUBO_SKIPPED_TOO_LARGE


def test_insufficient_penalty_is_detected_not_fixed(tiny_instance, test_config):
    """penalty 가 실제로 부족하면 PENALTY_INSUFFICIENT 로 기록하고 자동 수정하지 않는다.

    penalty 계수를 인위로 0 에 가깝게 줄이면 ground state 가 capacity constraint 를
    위반하게 된다(시설을 하나도 열지 않는 쪽이 energy 가 낮아진다).
    """
    from cflp_bd_qa_ex2.qubo.builder import build_master_qubo as _build

    qubo = _build(tiny_instance, [], test_config)

    # 원래 penalty 로는 충분해야 한다.
    baseline = validate_qubo_exactly(qubo, tiny_instance, [], test_config)
    assert baseline.penalty_status is Status.SUCCESS

    # penalty 항만 사실상 제거한 QUBO 를 직접 만든다.
    weak = _build(tiny_instance, [], test_config)
    scale = 1e-9
    residual = weak.residual_forms["capacity"]
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    offset = residual.square_into(linear, quadratic, weak.penalty.penalty_alpha_capacity)

    # 기존 penalty 기여분을 빼고 아주 작은 가중치로 다시 더한다.
    for name, coefficient in linear.items():
        weak.linear[name] = weak.linear.get(name, 0.0) - coefficient
    for key, coefficient in quadratic.items():
        weak.quadratic[key] = weak.quadratic.get(key, 0.0) - coefficient
    weak.offset -= offset

    weak_linear: dict[str, float] = {}
    weak_quadratic: dict[tuple[str, str], float] = {}
    weak.offset += residual.square_into(
        weak_linear, weak_quadratic, weak.penalty.penalty_alpha_capacity * scale
    )
    for name, coefficient in weak_linear.items():
        weak.linear[name] = weak.linear.get(name, 0.0) + coefficient
    for key, coefficient in weak_quadratic.items():
        weak.quadratic[key] = weak.quadratic.get(key, 0.0) + coefficient

    result = validate_qubo_exactly(weak, tiny_instance, [], test_config)
    assert result.status is Status.EXACT_QUBO_VALIDATED
    assert result.penalty_status is Status.PENALTY_INSUFFICIENT
    assert result.fields["ground_state_mp_feasible"] is False
    # 자동 수정하지 않는다: penalty 계수는 그대로여야 한다.
    assert weak.penalty.penalty_alpha_capacity == pytest.approx(
        qubo.penalty.penalty_alpha_capacity
    )
