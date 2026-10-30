"""theta bound 를 적용한 QUBO 생성 테스트 (묶음 B).

핵심은 **기존 호출의 QUBO 가 완전히 동일하게 유지되는 것**이다.
bound 인자는 선택이며, 생략하면 baseline(S0) 과 비트 단위로 같아야 한다.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from cflp_bd_qa_ex2.benders.cuts import build_optimality_cut
from cflp_bd_qa_ex2.encoding.bounds import build_all_bounds
from cflp_bd_qa_ex2.encoding.slack import CutSlackRangeInvalid
from cflp_bd_qa_ex2.evaluation.bound_analysis import CutRecord, analyze_configuration, cut_slack_bits
from cflp_bd_qa_ex2.models.master import mp_objective, mp_violations
from cflp_bd_qa_ex2.models.subproblem import solve_subproblem
from cflp_bd_qa_ex2.qubo.builder import ROLE_Y, build_master_qubo


def fingerprint(qubo) -> str:
    """QUBO 를 계수 단위로 식별하는 해시."""
    payload = {
        "linear": sorted((k, round(v, 9)) for k, v in qubo.linear.items()),
        "quadratic": sorted((a, b, round(v, 9)) for (a, b), v in qubo.quadratic.items()),
        "offset": round(qubo.offset, 9),
        "labels": qubo.variable_labels,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cuts_for(inst, cfg, count=2):
    """y 를 조금씩 바꿔 가며 실제 dual 로 cut 을 만든다."""
    cuts = []
    for index in range(count):
        y = np.ones(inst.n_facilities, dtype=int)
        if index and inst.n_facilities > index:
            y[index - 1] = 0
            if float(inst.capacity @ y) < inst.total_demand:
                y[index - 1] = 1
        outcome, dual = solve_subproblem(inst, y, cfg)
        if dual is None:
            continue
        cuts.append(
            build_optimality_cut(index, inst.capacity.astype(float), dual.u, dual.v, index + 1)
        )
    return cuts


def test_default_call_is_bitwise_identical(small_instance, pilot_config):
    """인자를 생략하면 기존 QUBO 와 완전히 동일하다 (regression)."""
    cuts = _cuts_for(small_instance, pilot_config)
    for cut_set in ([], cuts):
        default = build_master_qubo(small_instance, cut_set, pilot_config)
        explicit = build_master_qubo(
            small_instance, cut_set, pilot_config,
            theta_lower_bound=0.0,
            theta_upper_bound=small_instance.objective_upper_bound,
            theta_bits=int(pilot_config["encoding"]["theta"]["bits"]),
        )
        assert fingerprint(default) == fingerprint(explicit)
        assert default.theta_encoding.theta_lower_bound == 0.0
        assert default.theta_encoding.theta_upper_bound == pytest.approx(
            small_instance.objective_upper_bound
        )


def test_s0_bound_reproduces_default(small_instance, pilot_config):
    """bound 전략 S0 로 명시해도 기존 QUBO 와 같다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    cuts = _cuts_for(small_instance, pilot_config)
    default = build_master_qubo(small_instance, cuts, pilot_config)
    with_s0 = build_master_qubo(
        small_instance, cuts, pilot_config,
        theta_lower_bound=bounds["S0"].theta_lower_bound,
        theta_upper_bound=bounds["S0"].theta_upper_bound,
    )
    assert fingerprint(default) == fingerprint(with_s0)


@pytest.mark.parametrize("strategy_id", ["S1", "S2", "S3"])
def test_tightened_bounds_change_encoding(strategy_id, small_instance, pilot_config):
    """S1~S3 는 구간이 좁아져 해상도가 좋아진다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    default = build_master_qubo(small_instance, [], pilot_config)
    tightened = build_master_qubo(
        small_instance, [], pilot_config,
        theta_lower_bound=bounds[strategy_id].theta_lower_bound,
        theta_upper_bound=bounds[strategy_id].theta_upper_bound,
    )
    assert tightened.theta_encoding.theta_delta < default.theta_encoding.theta_delta
    assert tightened.theta_encoding.theta_lower_bound == pytest.approx(
        bounds[strategy_id].theta_lower_bound
    )
    # theta bit 수는 그대로다 (E1-A 조건)
    assert tightened.theta_encoding.theta_bits == default.theta_encoding.theta_bits


@pytest.mark.parametrize("strategy_id", ["S0", "S1", "S2", "S3"])
def test_decoded_objective_matches_original_mp(strategy_id, small_instance, pilot_config):
    """디코딩한 (y, theta) 의 MP 목적값이 원래 식과 일치한다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    cuts = _cuts_for(small_instance, pilot_config)
    qubo = build_master_qubo(
        small_instance, cuts, pilot_config,
        theta_lower_bound=bounds[strategy_id].theta_lower_bound,
        theta_upper_bound=bounds[strategy_id].theta_upper_bound,
    )

    rng = np.random.default_rng(0)
    for _ in range(10):
        assignment = {v: int(rng.integers(0, 2)) for v in qubo.variable_labels}
        y = np.array(
            [assignment[n] for n in qubo.variables_by_role(ROLE_Y)], dtype=int
        )
        theta = qubo.theta_encoding.decode(
            {n: assignment[n] for n in qubo.theta_encoding.encoding.variable_names()}
        )
        # theta 는 항상 [theta_L, theta_U] 안에 있다
        assert bounds[strategy_id].theta_lower_bound - 1e-9 <= theta
        assert theta <= bounds[strategy_id].theta_upper_bound + 1e-9
        # 실제 QUBO objective의 상수항까지 원래 식과 대조한다.
        assert qubo.objective_form.evaluate(assignment) == pytest.approx(
            float(small_instance.fixed_cost @ y) + theta
        )
        assert mp_objective(small_instance, y, theta) == pytest.approx(
            float(small_instance.fixed_cost @ y) + theta
        )


@pytest.mark.parametrize("strategy_id", ["S0", "S1", "S2", "S3"])
def test_cut_residual_matches_original_inequality(strategy_id, small_instance, pilot_config):
    """QUBO 의 cut residual 이 원래 MP 부등식과 부호까지 일치한다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    cuts = _cuts_for(small_instance, pilot_config)
    if not cuts:
        pytest.skip("cut 을 만들지 못했다")

    qubo = build_master_qubo(
        small_instance, cuts, pilot_config,
        theta_lower_bound=bounds[strategy_id].theta_lower_bound,
        theta_upper_bound=bounds[strategy_id].theta_upper_bound,
    )

    rng = np.random.default_rng(1)
    for _ in range(10):
        assignment = {v: int(rng.integers(0, 2)) for v in qubo.variable_labels}
        y = np.array([assignment[n] for n in qubo.variables_by_role(ROLE_Y)], dtype=int)
        theta = qubo.theta_encoding.decode(
            {n: assignment[n] for n in qubo.theta_encoding.encoding.variable_names()}
        )

        for cut in cuts:
            form = qubo.residual_forms[f"cut_{cut.index}"]
            # residual = theta + sum s_j v_j y_j - sum u_i - q_k
            slack_names = [
                n for n in qubo.cut_slacks[cut.index].encoding.variable_names()
            ]
            q_k = qubo.cut_slacks[cut.index].decode(
                {n: assignment[n] for n in slack_names}
            )
            expected = cut.residual(y, theta) - q_k
            assert form.evaluate(assignment) == pytest.approx(expected, abs=1e-6)

        # 원래 부등식 기준의 위반량도 bound 와 무관하게 같은 식이어야 한다
        violations = mp_violations(small_instance, y, theta, cuts)
        assert violations["max_mp_violation"] >= 0.0


def test_offline_slack_bits_match_builder(small_instance, pilot_config):
    """offline 분석의 cut slack bit 가 실제 builder 와 같다."""
    bounds = build_all_bounds(small_instance, pilot_config)
    cuts = _cuts_for(small_instance, pilot_config)
    if not cuts:
        pytest.skip("cut 을 만들지 못했다")

    for strategy_id in ("S0", "S1", "S2", "S3"):
        bound = bounds[strategy_id]
        qubo = build_master_qubo(
            small_instance, cuts, pilot_config,
            theta_lower_bound=bound.theta_lower_bound,
            theta_upper_bound=bound.theta_upper_bound,
        )
        records = [
            CutRecord(iteration=c.index, dual_u=c.u, dual_v=c.v, activated=True)
            for c in cuts
        ]
        analysis = analyze_configuration(
            small_instance, bound, records,
            theta_bits=int(pilot_config["encoding"]["theta"]["bits"]),
        )
        assert analysis["logical_variables"] == qubo.num_variables
        assert analysis["logical_edges"] == len(qubo.quadratic)
        assert analysis["cut_slack_bits_total"] == sum(
            s.cut_slack_bits for s in qubo.cut_slacks
        )


def test_invalid_slack_range_raises_like_builder(small_instance):
    """허용 오차를 넘는 음수 S_k_max 는 offline 분석도 예외를 낸다."""
    capacity = small_instance.capacity.astype(float)
    # A_k 를 theta_U 보다 크게 만들어 S_k_max 를 음수로 만든다
    record = CutRecord(
        iteration=1,
        dual_u=np.full(1, 1000.0),
        dual_v=np.zeros(small_instance.n_facilities),
        activated=True,
    )
    with pytest.raises(CutSlackRangeInvalid):
        cut_slack_bits(record, capacity, theta_upper_bound=10.0, delta_cut=1.0)

    # 허용 오차 안의 음수는 0 으로 보고 통과한다
    bits, raw_max, _ = cut_slack_bits(
        record, capacity, theta_upper_bound=1000.0 - 1e-9, delta_cut=1.0
    )
    assert bits == 0 and raw_max == 0.0
