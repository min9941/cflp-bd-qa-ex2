"""Original MS-CFLP 모델 (ground truth).

    min  sum_j f_j y_j + sum_i sum_j d_i c_ij x_ij
    s.t. sum_j x_ij = 1                    for all i
         sum_i d_i x_ij <= s_j y_j         for all j
         y_j in {0,1},  x_ij >= 0

``x_ij <= y_j`` 는 baseline 에 포함하지 않는다. d_i > 0 이고 capacity constraint 가
있으므로 y_j = 0 이면 sum_i d_i x_ij <= 0 에서 자동으로 x_ij = 0 이 된다.
또한 현재 dual formulation 에 해당 제약의 dual 변수가 없다.
"""

from __future__ import annotations

import numpy as np

from ..data.generator import Instance
from ..solvers.gurobi import (
    GRB,
    SolveOutcome,
    Status,
    classify_gurobi_error,
    default_params,
    gurobi_env,
    map_model_status,
    timed,
)
from ..solvers.gurobi import gp


def solve_original_cflp(inst: Instance, cfg: dict) -> SolveOutcome:
    """Original MS-CFLP 를 Gurobi 로 직접 푼다.

    Args:
        inst: 대상 인스턴스.
        cfg: 전체 설정 dict.

    Returns:
        :class:`SolveOutcome`. ``status`` 가 ``OPTIMAL`` 인 경우에만
        ``objective`` 를 ground truth 로 사용할 수 있다. Time limit 으로 얻은
        incumbent 는 ground truth 로 부르지 않는다.
    """
    nf, nc = inst.n_facilities, inst.n_customers
    try:
        with timed() as elapsed, gurobi_env(default_params(cfg)) as env:
            model = gp.Model("original_cflp", env=env)
            y = model.addVars(nf, vtype=GRB.BINARY, name="y")
            x = model.addVars(nc, nf, lb=0.0, name="x")

            model.setObjective(
                gp.quicksum(float(inst.fixed_cost[j]) * y[j] for j in range(nf))
                + gp.quicksum(
                    float(inst.demand[i] * inst.transport_cost[i, j]) * x[i, j]
                    for i in range(nc)
                    for j in range(nf)
                ),
                GRB.MINIMIZE,
            )
            for i in range(nc):
                model.addConstr(gp.quicksum(x[i, j] for j in range(nf)) == 1.0, name=f"assign_{i}")
            for j in range(nf):
                model.addConstr(
                    gp.quicksum(float(inst.demand[i]) * x[i, j] for i in range(nc))
                    <= float(inst.capacity[j]) * y[j],
                    name=f"capacity_{j}",
                )
            model.optimize()

            status = map_model_status(model.Status)
            payload: dict = {"gurobi_status": int(model.Status)}
            objective = bound = None
            if model.SolCount > 0:
                objective = float(model.ObjVal)
                payload["y"] = np.array(
                    [int(round(y[j].X)) for j in range(nf)], dtype=int
                ).tolist()
                payload["x"] = np.array(
                    [[float(x[i, j].X) for j in range(nf)] for i in range(nc)]
                ).tolist()
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


def evaluate_original_objective(
    inst: Instance, y: np.ndarray, sp_value: float
) -> float:
    """주어진 y 와 SP 값으로 original CFLP 목적값을 계산한다.

    ``Obj = sum_j f_j y_j + Q(y)``. Q(y) 는 SP 에서 얻은 continuous x 로 계산된
    값이어야 한다.
    """
    return float(inst.fixed_cost @ np.asarray(y, dtype=float)) + float(sp_value)
