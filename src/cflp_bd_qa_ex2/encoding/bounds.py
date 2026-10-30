"""Structure-based theta bound 전략 (S0~S3).

Benders recourse 변수 theta 의 encoding 구간 ``[theta_L, theta_U]`` 를
어떻게 잡느냐에 따라 해상도와 QUBO 크기가 달라진다.

| ID | 이름 | theta_L | theta_U |
|----|------|---------|---------|
| S0 | Baseline | 0 | sum_j f_j + sum_i sum_j d_i c_ij |
| S1 | Cost-Based | 0 | sum_i d_i max_j c_ij |
| S2 | Assignment Relaxation | sum_i d_i min_j c_ij | sum_i d_i max_j c_ij |
| S3 | All-Open LP | Q(1) | sum_i d_i max_j c_ij |

**유효성 증명**

theta 는 recourse ``Q(y)`` 를 나타낸다. 모든 feasible y 에 대해:

- 상한: ``sum_j x_ij = 1``, ``x_ij >= 0`` 이므로
  ``sum_j d_i c_ij x_ij <= d_i max_j c_ij``. 고객 전체로 합하면 S1~S3 의 상한.
  S0 는 여기에 fixed cost 까지 더한 더 느슨한 값이다.
- 하한(S2): capacity 를 무시한 배정이 비용을 더 낮출 수 없으므로
  ``Q(y) >= sum_i d_i min_j c_ij``.
- 하한(S3): ``X(y) ⊆ X(1)`` 이므로 ``Q(y) >= Q(1)``.
  시설을 덜 열면 선택지가 줄어 수송비가 올라간다.
  S2 의 하한보다 조이거나 같다: ``sum_i d_i min_j c_ij <= Q(1)``.

**주의.** 하한을 올리면 ``delta_theta = (theta_U - theta_L)/(2^b - 1)`` 가 작아지고,
``delta_cut = r_cut * delta_theta`` 도 작아진다. 반면 cut slack 범위
``S_k_max = theta_U - A_k`` 는 theta_L 과 무관하다. 따라서 **같은 cut pool 에서
S1 -> S2 -> S3 의 하한 강화는 cut slack bit 수를 줄이지 못한다**(증가 또는 유지).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.generator import Instance
from ..status import Status

STRATEGY_IDS: tuple[str, ...] = ("S0", "S1", "S2", "S3")

STRATEGY_NAMES: dict[str, str] = {
    "S0": "Baseline",
    "S1": "Cost-Based",
    "S2": "Assignment Relaxation",
    "S3": "All-Open LP",
}


class BoundError(ValueError):
    """Bound 계산이 실패했거나 유효하지 않을 때 발생하는 예외."""


@dataclass(frozen=True)
class ThetaBound:
    """하나의 bound 전략이 만든 theta 구간."""

    strategy_id: str
    strategy_name: str
    instance_id: str
    theta_lower_bound: float
    theta_upper_bound: float
    computation_time: float
    requires_lp: bool
    lp_status: str | None = None
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    @property
    def theta_range(self) -> float:
        """구간 폭."""
        return self.theta_upper_bound - self.theta_lower_bound

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        return {
            "bound_strategy": self.strategy_id,
            "bound_strategy_name": self.strategy_name,
            "instance_id": self.instance_id,
            "theta_lower_bound": self.theta_lower_bound,
            "theta_upper_bound": self.theta_upper_bound,
            "theta_range": self.theta_range,
            "bound_computation_time": self.computation_time,
            "bound_requires_lp": self.requires_lp,
            "bound_lp_status": self.lp_status,
        }


def upper_bound_baseline(inst: Instance) -> float:
    """S0: ``sum_j f_j + sum_i sum_j c_ij d_i`` (기존 U_obj)."""
    return float(inst.objective_upper_bound)


def upper_bound_cost_based(inst: Instance) -> float:
    """S1~S3: ``sum_i d_i max_j c_ij``.

    고객마다 가장 비싼 시설로 전량 보내는 경우이며, 수송비만 담는다.
    fixed cost 를 포함하지 않으므로 S0 보다 훨씬 작다.
    """
    return float((inst.demand.astype(float) * inst.transport_cost.max(axis=1)).sum())


def lower_bound_assignment_relaxation(inst: Instance) -> float:
    """S2: ``sum_i d_i min_j c_ij``. capacity 를 무시한 배정 비용."""
    return float((inst.demand.astype(float) * inst.transport_cost.min(axis=1)).sum())


def lower_bound_all_open_lp(inst: Instance, cfg: dict) -> tuple[float, str]:
    """S3: ``Q(1)``. 모든 시설을 연 transportation problem 의 최적값.

    MIP 가 아니라 LP 이므로 제한 라이선스에서도 풀린다
    (25x50 이면 변수 1250개).

    Returns:
        ``(Q(1), gurobi_status_name)``.

    Raises:
        BoundError: LP 가 OPTIMAL 로 끝나지 않은 경우. 값을 추측하지 않는다.
    """
    from ..models.subproblem import solve_subproblem

    all_open = np.ones(inst.n_facilities, dtype=int)
    outcome, _ = solve_subproblem(inst, all_open, cfg)
    if outcome.status is not Status.OPTIMAL or outcome.objective is None:
        raise BoundError(
            f"{inst.instance_id}: Q(1) LP 가 OPTIMAL 이 아니다 ({outcome.status}). "
            f"{outcome.message}"
        )
    return float(outcome.objective), str(outcome.status)


def build_bound(strategy_id: str, inst: Instance, cfg: dict) -> ThetaBound:
    """전략 ID 로 theta bound 를 계산한다.

    Args:
        strategy_id: ``"S0"`` ~ ``"S3"``.
        inst: 대상 인스턴스.
        cfg: 전체 설정 dict (S3 의 LP 에 필요).

    Returns:
        계산된 :class:`ThetaBound`.

    Raises:
        BoundError: 알 수 없는 전략이거나 구간이 유효하지 않은 경우.
    """
    import time

    if strategy_id not in STRATEGY_IDS:
        raise BoundError(f"알 수 없는 bound 전략: {strategy_id!r}. 가능: {STRATEGY_IDS}")

    start = time.perf_counter()
    lp_status: str | None = None
    requires_lp = False

    if strategy_id == "S0":
        lower, upper = 0.0, upper_bound_baseline(inst)
    elif strategy_id == "S1":
        lower, upper = 0.0, upper_bound_cost_based(inst)
    elif strategy_id == "S2":
        lower, upper = lower_bound_assignment_relaxation(inst), upper_bound_cost_based(inst)
    else:  # S3
        requires_lp = True
        lower, lp_status = lower_bound_all_open_lp(inst, cfg)
        upper = upper_bound_cost_based(inst)

    elapsed = time.perf_counter() - start
    if upper <= lower:
        raise BoundError(
            f"{inst.instance_id}/{strategy_id}: theta 구간이 유효하지 않다 "
            f"[{lower}, {upper}] (THETA_RANGE_INVALID)"
        )

    return ThetaBound(
        strategy_id=strategy_id,
        strategy_name=STRATEGY_NAMES[strategy_id],
        instance_id=inst.instance_id,
        theta_lower_bound=lower,
        theta_upper_bound=upper,
        computation_time=elapsed,
        requires_lp=requires_lp,
        lp_status=lp_status,
        metadata={},
    )


def build_all_bounds(inst: Instance, cfg: dict) -> dict[str, ThetaBound]:
    """S0~S3 를 모두 계산한다."""
    return {sid: build_bound(sid, inst, cfg) for sid in STRATEGY_IDS}


def validate_bound(
    bound: ThetaBound,
    recourse_at_optimum: float,
    tolerance: float = 1e-6,
    reference_kind: str = "optimal",
) -> dict[str, Any]:
    """최적해의 ``Q(y*)`` 가 구간 안에 있는지 확인한다.

    **이것은 증명이 아니라 구현 검증이다.** 이론적 유효성은 feasible region
    포함 관계로 이미 증명된다. 여기서는 부동소수 오차나 구현 오류를 잡는다.

    Args:
        bound: 검사할 bound.
        recourse_at_optimum: 기준 해에서 SP 를 푼 값 Q(y).
        tolerance: 상대 허용 오차.
        reference_kind: ``"optimal"`` 이면 증명된 최적해, ``"incumbent"`` 면
            최적성이 증명되지 않은 해이다. 후자는 검증 강도가 약하므로
            결과에 그대로 표시한다.

    Returns:
        ``lower_violated``, ``upper_violated``, ``clipping_count`` 등 진단 dict.
        위반이 있어도 예외를 던지지 않는다. 결과에서 제외하지 않고 기록한다.
        기준값이 유한하지 않으면 ``bound_valid`` 가 ``None`` 이다
        (검사하지 못한 것을 "통과" 로 기록하지 않는다).
    """
    if recourse_at_optimum is None or not math.isfinite(float(recourse_at_optimum)):
        return {
            "bound_strategy": bound.strategy_id,
            "instance_id": bound.instance_id,
            "reference_kind": reference_kind,
            "recourse_at_reference": recourse_at_optimum,
            "theta_lower_bound": bound.theta_lower_bound,
            "theta_upper_bound": bound.theta_upper_bound,
            "lower_slack": None,
            "upper_slack": None,
            "lower_violated": None,
            "upper_violated": None,
            "theta_clipping_count": None,
            "bound_valid": None,
            "validation_status": "not_checked_non_finite_reference",
        }

    recourse_at_optimum = float(recourse_at_optimum)
    scale = max(1.0, abs(recourse_at_optimum))
    lower_slack = recourse_at_optimum - bound.theta_lower_bound
    upper_slack = bound.theta_upper_bound - recourse_at_optimum
    lower_violated = lower_slack < -tolerance * scale
    upper_violated = upper_slack < -tolerance * scale

    return {
        "bound_strategy": bound.strategy_id,
        "instance_id": bound.instance_id,
        "reference_kind": reference_kind,
        "recourse_at_reference": recourse_at_optimum,
        "theta_lower_bound": bound.theta_lower_bound,
        "theta_upper_bound": bound.theta_upper_bound,
        "lower_slack": lower_slack,
        "upper_slack": upper_slack,
        "lower_violated": bool(lower_violated),
        "upper_violated": bool(upper_violated),
        "theta_clipping_count": int(lower_violated) + int(upper_violated),
        "bound_valid": not (lower_violated or upper_violated),
        "validation_status": "checked",
    }
