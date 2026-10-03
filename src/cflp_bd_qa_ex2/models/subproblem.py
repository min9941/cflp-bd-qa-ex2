"""Benders subproblem (primal / dual) 과 dual minimality guard.

Primal SP (주어진 ybar 에 대해):

    min  sum_i sum_j d_i c_ij x_ij
    s.t. sum_j x_ij = 1               for all i     -> dual u_i (free)
         sum_i d_i x_ij <= s_j ybar_j for all j     -> dual v_j (>= 0)
         x_ij >= 0

Dual SP:

    max  sum_i u_i - sum_j s_j ybar_j v_j
    s.t. u_i - d_i v_j <= d_i c_ij
         v_j >= 0,  u_i free

모든 customer-facility arc 가 존재하고 x 가 continuous 이므로
aggregate capacity constraint 가 만족되면 SP 는 이론적으로 feasible 하다.
SP 가 INFEASIBLE 이면 feasibility cut 을 임의로 만들지 않고 중단한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.generator import Instance
from ..status import DualGuardAction, DualMinimalityStatus, Status
from ..solvers.gurobi import (
    GRB,
    SolveOutcome,
    classify_gurobi_error,
    default_params,
    gp,
    gurobi_env,
    map_model_status,
    timed,
)


@dataclass
class DualSolution:
    """SP 의 dual 해와 그 진단값."""

    u: np.ndarray
    v: np.ndarray
    dual_objective: float
    primal_objective: float
    strong_duality_gap: float
    dual_constraint_max_violation: float
    projected_v: np.ndarray
    minimality_status: DualMinimalityStatus = DualMinimalityStatus.NOT_CHECKED
    v_minimality_abs_gap: float = 0.0
    v_minimality_rel_gap: float = 0.0
    v_minimality_open_gap: float = 0.0
    v_minimality_closed_gap: float = 0.0
    violating_facilities: list[int] = field(default_factory=list)

    @property
    def dual_vector_hash(self) -> str:
        """(u, v) 의 결정적 해시. dual representative 추적에 사용한다."""
        blob = np.concatenate([np.round(self.u, 12), np.round(self.v, 12)]).tobytes()
        return hashlib.sha256(blob).hexdigest()[:32]

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        return {
            "raw_dual_u": self.u.tolist(),
            "raw_dual_v": self.v.tolist(),
            "projected_dual_v": self.projected_v.tolist(),
            "dual_objective": self.dual_objective,
            "primal_subproblem_objective": self.primal_objective,
            "strong_duality_gap": self.strong_duality_gap,
            "dual_constraint_max_violation": self.dual_constraint_max_violation,
            "v_minimality_abs_gap": self.v_minimality_abs_gap,
            "v_minimality_rel_gap": self.v_minimality_rel_gap,
            "v_minimality_open_gap": self.v_minimality_open_gap,
            "v_minimality_closed_gap": self.v_minimality_closed_gap,
            "dual_vector_hash": self.dual_vector_hash,
            "dual_minimality_status": str(self.minimality_status),
        }


def compute_minimal_v(u: np.ndarray, demand: np.ndarray, transport_cost: np.ndarray) -> np.ndarray:
    """주어진 u 에 대해 componentwise 최소인 feasible v 를 계산한다.

    dual constraint ``u_i - d_i v_j <= d_i c_ij`` 로부터
    ``v_j >= u_i / d_i - c_ij`` 이므로 최소값은
    ``max(0, max_i (u_i / d_i - c_ij))`` 이다.

    주의: 이것은 "minimum-norm dual" 이 아니다. u 자체도 비유일할 수 있으므로
    이 투영은 **반환된 u 를 고정했을 때의** componentwise 최소성만 보장한다.

    Args:
        u: assignment constraint 의 dual, shape ``(n_customers,)``.
        demand: d_i, shape ``(n_customers,)``.
        transport_cost: c_ij, shape ``(n_customers, n_facilities)``.

    Returns:
        componentwise 최소 v, shape ``(n_facilities,)``.
    """
    ratio = u[:, None] / demand[:, None].astype(float)
    return np.maximum(0.0, (ratio - transport_cost).max(axis=0))


def check_dual_minimality(
    dual: DualSolution,
    y: np.ndarray,
    atol: float,
    rtol: float,
) -> DualSolution:
    """Dual minimality guard 를 적용한다(자동 수정은 하지 않는다).

    판정 기준은 대칭 tolerance 이다.

        |v_j - v_j_min| <= atol + rtol * max(|v_j|, |v_j_min|)

    닫힌 시설(y_j = 0)의 v_j 는 dual objective 계수가 0이므로 optimal face 위에서
    위로 무한히 커질 수 있다. 그 팽창은 cut coefficient 와 cut slack bit 수를
    바꾸므로 재현성을 해친다. 이 guard 는 그것을 **탐지만** 한다.

    열린 시설(y_j = 1)에서는 strong duality 상 v_j 가 이미 최소여야 하므로,
    열린 시설에서 guard 가 걸리면 dual 비유일성이 아니라 수치 오류 신호이다.

    Args:
        dual: 검사할 dual 해.
        y: 현재 ybar.
        atol: 절대 허용 오차.
        rtol: 상대 허용 오차.

    Returns:
        진단 field 가 채워진 동일 객체.
    """
    gap = np.abs(dual.v - dual.projected_v)
    tol = atol + rtol * np.maximum(np.abs(dual.v), np.abs(dual.projected_v))
    violating = np.where(gap > tol)[0]

    open_mask = np.asarray(y, dtype=int) == 1
    closed_mask = ~open_mask

    dual.v_minimality_abs_gap = float(gap.max()) if gap.size else 0.0
    denom = np.maximum(1e-12, np.maximum(np.abs(dual.v), np.abs(dual.projected_v)))
    dual.v_minimality_rel_gap = float((gap / denom).max()) if gap.size else 0.0
    dual.v_minimality_open_gap = float(gap[open_mask].max()) if open_mask.any() else 0.0
    dual.v_minimality_closed_gap = float(gap[closed_mask].max()) if closed_mask.any() else 0.0
    dual.violating_facilities = [int(j) for j in violating]
    dual.minimality_status = (
        DualMinimalityStatus.DUAL_NOT_MINIMAL
        if violating.size > 0
        else DualMinimalityStatus.MINIMAL
    )
    return dual


def dual_continuation_allowed(
    sp_status: Status,
    dual: DualSolution,
    duality_tolerance: float,
    dual_feasibility_tolerance: float,
) -> tuple[bool, str]:
    """Guard 위반 시 계속 진행해도 되는지 판정한다.

    cut 이 dual-feasible 하다는 것만으로는 부족하다. 다음이 모두 통과해야 한다.

    1. SP status 가 ``OPTIMAL``
    2. dual constraint 가 feasible
    3. primal-dual gap 이 허용 오차 이내

    Returns:
        ``(allowed, reason)``.
    """
    if sp_status is not Status.OPTIMAL:
        return False, f"SP status 가 OPTIMAL 이 아니다: {sp_status}"
    if dual.dual_constraint_max_violation > dual_feasibility_tolerance:
        return False, (
            f"dual constraint 위반 {dual.dual_constraint_max_violation:.3e} "
            f"> {dual_feasibility_tolerance:.3e}"
        )
    scale = max(1.0, abs(dual.primal_objective))
    if dual.strong_duality_gap / scale > duality_tolerance:
        return False, (
            f"strong duality gap {dual.strong_duality_gap:.3e} 가 허용 오차를 넘는다"
        )
    return True, "SP optimality / dual feasibility / strong duality 모두 통과"


def solve_subproblem(
    inst: Instance,
    y: np.ndarray,
    cfg: dict,
) -> tuple[SolveOutcome, DualSolution | None]:
    """주어진 ybar 에 대해 primal SP 를 풀고 dual 을 추출한다.

    Args:
        inst: 대상 인스턴스.
        y: 현재 ybar (0/1 배열).
        cfg: 전체 설정 dict.

    Returns:
        ``(outcome, dual)``. SP 가 OPTIMAL 이 아니면 ``dual`` 은 ``None`` 이다.
    """
    nf, nc = inst.n_facilities, inst.n_customers
    y = np.asarray(y, dtype=int)
    try:
        with timed() as elapsed, gurobi_env(default_params(cfg)) as env:
            model = gp.Model("subproblem", env=env)
            x = model.addVars(nc, nf, lb=0.0, name="x")
            model.setObjective(
                gp.quicksum(
                    float(inst.demand[i] * inst.transport_cost[i, j]) * x[i, j]
                    for i in range(nc)
                    for j in range(nf)
                ),
                GRB.MINIMIZE,
            )
            assign = {
                i: model.addConstr(gp.quicksum(x[i, j] for j in range(nf)) == 1.0)
                for i in range(nc)
            }
            capacity = {
                j: model.addConstr(
                    gp.quicksum(float(inst.demand[i]) * x[i, j] for i in range(nc))
                    <= float(inst.capacity[j]) * float(y[j])
                )
                for j in range(nf)
            }
            model.optimize()
            status = map_model_status(model.Status)

            if status is not Status.OPTIMAL:
                return (
                    SolveOutcome(
                        status=status,
                        runtime=elapsed[0],
                        gurobi_status=int(model.Status),
                        message="SP 가 OPTIMAL 이 아니다. feasibility cut 을 임의 생성하지 않는다",
                    ),
                    None,
                )

            primal_objective = float(model.ObjVal)
            u = np.array([assign[i].Pi for i in range(nc)], dtype=float)
            # capacity constraint 는 <= 이므로 Gurobi 의 Pi 는 음수 또는 0이다.
            # dual formulation 의 v_j >= 0 에 맞추기 위해 부호를 뒤집는다.
            v = np.array([-capacity[j].Pi for j in range(nf)], dtype=float)
            x_value = np.array(
                [[float(x[i, j].X) for j in range(nf)] for i in range(nc)], dtype=float
            )
    except Exception as exc:
        status, message = classify_gurobi_error(exc)
        return SolveOutcome(status=status, message=message), None

    v = np.maximum(v, 0.0)  # 수치 오차로 생긴 -0.0 정리 (부호 규약 유지)
    dual_objective = float(u.sum() - (inst.capacity.astype(float) * y) @ v)
    # dual constraint 위반: u_i - d_i v_j - d_i c_ij <= 0
    lhs = u[:, None] - inst.demand[:, None].astype(float) * v[None, :]
    rhs = inst.demand[:, None].astype(float) * inst.transport_cost
    violation = float(np.maximum(0.0, lhs - rhs).max())

    dual = DualSolution(
        u=u,
        v=v,
        dual_objective=dual_objective,
        primal_objective=primal_objective,
        strong_duality_gap=abs(primal_objective - dual_objective),
        dual_constraint_max_violation=violation,
        projected_v=compute_minimal_v(u, inst.demand, inst.transport_cost),
    )

    outcome = SolveOutcome(
        status=Status.OPTIMAL,
        objective=primal_objective,
        runtime=elapsed[0],
        payload={"x": x_value.tolist()},
    )
    return outcome, dual
