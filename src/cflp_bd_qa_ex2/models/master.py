"""Master Problem 의 세 계층.

명세 6절에 따라 서로 다른 세 모델을 분리한다.

- ``continuous_mp``  : y in {0,1}, theta in R. **유효한 Benders lower bound** 와
                       종료 판정에 쓰는 유일한 모델.
- ``encoded_mp``     : theta 를 binary encoding 하되 원래 MP inequality 를 그대로
                       적용. 순수한 discretization 영향을 측정한다.
- ``penalized_qubo`` : slack 과 constraint penalty 가 들어간 실제 QUBO
                       (qubo/builder.py 에서 생성).

MP formulation:

    min  sum_j f_j y_j + theta
    s.t. sum_j s_j y_j >= D
         theta + sum_j s_j v_j^k y_j >= sum_i u_i^k     for k = 1..K
         y_j in {0,1}, theta >= 0
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..benders.cuts import OptimalityCut
from ..data.generator import Instance
from ..encoding.theta import ThetaEncoding
from ..status import Status
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


def solve_continuous_mp(
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    cfg: dict,
) -> SolveOutcome:
    """Continuous-theta MP 를 Gurobi 로 푼다.

    이 모델의 최적값(또는 유효한 ``ObjBound``)만이 certified lower bound 로
    사용될 수 있다. SA/QA 의 목적값은 lower bound 로 쓰지 않는다.

    Args:
        inst: 대상 인스턴스.
        cuts: 현재까지의 optimality cut 목록.
        cfg: 전체 설정 dict.

    Returns:
        :class:`SolveOutcome`. ``payload["y"]``, ``payload["theta"]`` 포함.
    """
    nf = inst.n_facilities
    try:
        with timed() as elapsed, gurobi_env(default_params(cfg)) as env:
            model = gp.Model("continuous_mp", env=env)
            y = model.addVars(nf, vtype=GRB.BINARY, name="y")
            theta = model.addVar(lb=0.0, name="theta")
            model.setObjective(
                gp.quicksum(float(inst.fixed_cost[j]) * y[j] for j in range(nf)) + theta,
                GRB.MINIMIZE,
            )
            model.addConstr(
                gp.quicksum(float(inst.capacity[j]) * y[j] for j in range(nf))
                >= float(inst.total_demand),
                name="aggregate_capacity",
            )
            for cut in cuts:
                model.addConstr(
                    theta
                    + gp.quicksum(float(cut.y_coefficients[j]) * y[j] for j in range(nf))
                    >= float(cut.rhs_constant),
                    name=f"cut_{cut.index}",
                )
            model.optimize()
            status = map_model_status(model.Status)
            payload: dict[str, Any] = {"gurobi_status": int(model.Status)}
            objective = bound = None
            if model.SolCount > 0:
                objective = float(model.ObjVal)
                payload["y"] = [int(round(y[j].X)) for j in range(nf)]
                payload["theta"] = float(theta.X)
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                bound = float(model.ObjBound)
    except Exception as exc:
        status, message = classify_gurobi_error(exc)
        return SolveOutcome(status=status, message=message)

    return SolveOutcome(
        status=status,
        objective=objective,
        bound=bound,
        runtime=elapsed[0],
        gurobi_status=payload.get("gurobi_status"),
        payload=payload,
    )


def solve_encoded_mp(
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    theta_encoding: ThetaEncoding,
    cfg: dict,
) -> SolveOutcome:
    """theta 를 binary encoding 한 MP 를 푼다 (inequality 는 원형 유지).

    penalty 나 slack 을 넣지 않으므로, continuous MP 와의 차이는 **순수하게
    theta discretization 때문**이다.
    """
    nf = inst.n_facilities
    enc = theta_encoding.encoding
    try:
        with timed() as elapsed, gurobi_env(default_params(cfg)) as env:
            model = gp.Model("encoded_mp", env=env)
            y = model.addVars(nf, vtype=GRB.BINARY, name="y")
            t = model.addVars(enc.n_bits, vtype=GRB.BINARY, name="t")
            theta_expr = gp.quicksum(float(coef) * t[p] for p, coef in enumerate(enc.coefficients()))
            model.setObjective(
                gp.quicksum(float(inst.fixed_cost[j]) * y[j] for j in range(nf)) + theta_expr,
                GRB.MINIMIZE,
            )
            model.addConstr(
                gp.quicksum(float(inst.capacity[j]) * y[j] for j in range(nf))
                >= float(inst.total_demand),
                name="aggregate_capacity",
            )
            for cut in cuts:
                model.addConstr(
                    theta_expr
                    + gp.quicksum(float(cut.y_coefficients[j]) * y[j] for j in range(nf))
                    >= float(cut.rhs_constant),
                    name=f"cut_{cut.index}",
                )
            model.optimize()
            status = map_model_status(model.Status)
            payload: dict[str, Any] = {"gurobi_status": int(model.Status)}
            objective = bound = None
            if model.SolCount > 0:
                objective = float(model.ObjVal)
                payload["y"] = [int(round(y[j].X)) for j in range(nf)]
                payload["theta"] = float(enc.decode([int(round(t[p].X)) for p in range(enc.n_bits)]))
                payload["theta_bits"] = [int(round(t[p].X)) for p in range(enc.n_bits)]
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                bound = float(model.ObjBound)
    except Exception as exc:
        status, message = classify_gurobi_error(exc)
        return SolveOutcome(status=status, message=message)

    return SolveOutcome(
        status=status,
        objective=objective,
        bound=bound,
        runtime=elapsed[0],
        gurobi_status=payload.get("gurobi_status"),
        payload=payload,
    )


def mp_violations(
    inst: Instance,
    y: np.ndarray,
    theta: float,
    cuts: Sequence[OptimalityCut],
) -> dict[str, float]:
    """원래 MP inequality 기준의 위반량을 계산한다 (명세 19절).

    auxiliary slack equality 의 exact residual 이 아니라 **원래 부등식**을
    기준으로 판정한다. dual 계수가 실수이므로 MP-feasible sample 이
    penalty residual 0 을 갖는다고 가정하지 않는다.

    Returns:
        ``capacity_violation``, ``max_cut_violation``, ``max_mp_violation`` dict.
    """
    y = np.asarray(y, dtype=float)
    capacity_violation = max(
        0.0, float(inst.total_demand) - float(inst.capacity.astype(float) @ y)
    )
    cut_violations = [cut.violation(y, theta) for cut in cuts]
    max_cut_violation = max(cut_violations) if cut_violations else 0.0
    return {
        "capacity_violation": capacity_violation,
        "max_cut_violation": max_cut_violation,
        "max_mp_violation": max(capacity_violation, max_cut_violation),
    }


def mp_objective(inst: Instance, y: np.ndarray, theta: float) -> float:
    """MP 목적값 ``sum_j f_j y_j + theta`` 를 계산한다."""
    return float(inst.fixed_cost @ np.asarray(y, dtype=float) + theta)
