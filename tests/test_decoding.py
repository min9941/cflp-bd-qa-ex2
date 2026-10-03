"""Sample 디코딩, MP feasibility, no-repair 정책 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from cflp_bd_qa_ex2.benders.cuts import build_optimality_cut
from cflp_bd_qa_ex2.models.master import mp_violations
from cflp_bd_qa_ex2.models.subproblem import solve_subproblem
from cflp_bd_qa_ex2.qubo.builder import ROLE_Y, build_master_qubo
from cflp_bd_qa_ex2.qubo.decode import (
    decode_sample,
    feasible_sample_rate,
    select_feasible_sample,
)
from cflp_bd_qa_ex2.solvers.sa import run_sa
from cflp_bd_qa_ex2.status import Status


def _assignment(qubo, y_values, theta_bits_on=0):
    """지정한 y 와 theta bit 로 배정을 만든다."""
    assignment = {v: 0 for v in qubo.variable_labels}
    for name, value in zip(qubo.variables_by_role(ROLE_Y), y_values):
        assignment[name] = int(value)
    for name in qubo.theta_encoding.encoding.variable_names()[:theta_bits_on]:
        assignment[name] = 1
    return assignment


def test_decode_recovers_y_and_theta(small_instance, test_config):
    """y 와 theta 가 original scale 로 복원된다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    assignment = _assignment(qubo, [1, 0, 1, 0], theta_bits_on=2)
    decoded = decode_sample(qubo, small_instance, [], assignment, 0.0, 0, test_config)

    assert decoded.decoded_y.tolist() == [1, 0, 1, 0]
    delta = qubo.theta_encoding.theta_delta
    assert decoded.decoded_theta == pytest.approx(delta * (1 + 2))


def test_mp_feasibility_uses_original_inequality(small_instance, test_config):
    """feasibility 는 slack equality residual 이 아니라 원래 부등식으로 본다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    assignment = _assignment(qubo, [1] * small_instance.n_facilities)
    # capacity slack bit 을 모두 0으로 두면 equality residual 은 0이 아니다.
    decoded = decode_sample(qubo, small_instance, [], assignment, 0.0, 0, test_config)
    assert decoded.penalty_residual > 0.0
    assert decoded.mp_feasible is True
    assert decoded.capacity_violation == 0.0


def test_capacity_violation_detected(small_instance, test_config):
    """용량이 부족한 y 는 infeasible 로 판정된다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    assignment = _assignment(qubo, [0] * small_instance.n_facilities)
    decoded = decode_sample(qubo, small_instance, [], assignment, 0.0, 0, test_config)
    assert decoded.mp_feasible is False
    assert decoded.capacity_violation == pytest.approx(float(small_instance.total_demand))


def test_cut_violation_detected(small_instance, test_config):
    """theta 가 너무 작으면 cut 위반으로 infeasible 하다."""
    y = np.ones(small_instance.n_facilities, dtype=int)
    _, dual = solve_subproblem(small_instance, y, test_config)
    cut = build_optimality_cut(0, small_instance.capacity.astype(float), dual.u, dual.v, 1)
    qubo = build_master_qubo(small_instance, [cut], test_config)

    assignment = _assignment(qubo, y, theta_bits_on=0)
    decoded = decode_sample(qubo, small_instance, [cut], assignment, 0.0, 0, test_config)
    assert decoded.max_cut_violation > 0.0
    assert decoded.mp_feasible is False


def test_selection_is_energy_ordered(small_instance, test_config):
    """energy 오름차순으로 검사하여 첫 feasible sample 을 고른다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    infeasible = _assignment(qubo, [0] * small_instance.n_facilities)
    feasible_a = _assignment(qubo, [1] * small_instance.n_facilities)
    feasible_b = _assignment(qubo, [1, 0, 0, 1])

    samples = [
        (infeasible, -10.0, 1, None),
        (feasible_a, 0.0, 1, None),
        (feasible_b, 5.0, 1, None),
    ]
    selected, decoded, status = select_feasible_sample(
        qubo, small_instance, [], samples, test_config
    )
    assert status is Status.SUCCESS
    assert selected.energy == 0.0
    assert selected.sample_rank == 1
    assert len(decoded) == 3


def test_no_feasible_sample_status(small_instance, test_config):
    """feasible sample 이 없으면 NO_FEASIBLE_SAMPLE 이다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    infeasible = _assignment(qubo, [0] * small_instance.n_facilities)
    selected, decoded, status = select_feasible_sample(
        qubo, small_instance, [], [(infeasible, 0.0, 1, None)], test_config
    )
    assert selected is None
    assert status is Status.NO_FEASIBLE_SAMPLE
    assert len(decoded) == 1


def test_no_repair_policy(small_instance, test_config):
    """infeasible sample 의 y 를 수정하지 않는다 (시설 추가/제거 금지)."""
    qubo = build_master_qubo(small_instance, [], test_config)
    y_values = [0] * small_instance.n_facilities
    assignment = _assignment(qubo, y_values)
    decoded = decode_sample(qubo, small_instance, [], assignment, 0.0, 0, test_config)
    assert decoded.decoded_y.tolist() == y_values  # 그대로 보존


def test_feasible_sample_rate_weighted():
    """feasible rate 는 occurrence 로 가중된다."""
    from cflp_bd_qa_ex2.qubo.decode import DecodedSample

    def make(feasible: bool, occurrences: int) -> DecodedSample:
        return DecodedSample(
            decoded_y=np.array([1]), decoded_theta=0.0, energy=0.0,
            decoded_mp_objective=0.0, capacity_violation=0.0, max_cut_violation=0.0,
            max_mp_violation=0.0, normalized_max_mp_violation=0.0, penalty_residual=0.0,
            mp_feasible=feasible, borderline_feasible=False, sample_rank=0,
            num_occurrences=occurrences,
        )

    assert feasible_sample_rate([make(True, 3), make(False, 1)]) == pytest.approx(0.75)


def test_sa_reaches_feasible_sample(small_instance, test_config):
    """SA sample 중 MP-feasible 한 것을 찾을 수 있다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    output = run_sa(qubo, test_config)
    assert output["samples"]
    energies = [energy for _, energy, _, _ in output["samples"]]
    assert energies == sorted(energies)

    selected, decoded, status = select_feasible_sample(
        qubo, small_instance, [], output["samples"], test_config
    )
    assert status is Status.SUCCESS
    assert selected.mp_feasible


def test_mp_violations_aggregate(small_instance, test_config):
    """max_mp_violation 은 capacity 와 cut 위반의 최대값이다."""
    y = np.zeros(small_instance.n_facilities, dtype=int)
    result = mp_violations(small_instance, y, 0.0, [])
    assert result["max_mp_violation"] == result["capacity_violation"]
