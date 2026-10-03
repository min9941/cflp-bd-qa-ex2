"""QUBO ground state 의 정확 계산과 penalty 충분성 검증.

두 경로를 지원한다.

- **exhaustive enumeration**: logical variable 수가
  ``validation.exhaustive_max_variables`` 이하일 때만 수행.
- **Gurobi MIQP**: 변수 수가 ``validation.miqp_max_variables`` 이하일 때 수행.
  제한 라이선스는 이차항이 있으면 200 변수까지만 허용하므로 이 값을 넘으면
  ``GUROBI_SIZE_LIMIT`` 이 된다.

초과 시 임의로 축소하지 않고 ``EXACT_QUBO_SKIPPED_TOO_LARGE`` 로 기록한다.

Penalty 가 부족하면 자동으로 값을 바꾸지 않고 ``PENALTY_INSUFFICIENT`` 로
기록한다.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..benders.cuts import OptimalityCut
from ..data.generator import Instance
from ..status import Status
from ..solvers.gurobi import (
    GRB,
    classify_gurobi_error,
    default_params,
    gp,
    gurobi_env,
    map_model_status,
)
from .builder import MasterQUBO
from .decode import decode_sample


@dataclass
class ExactResult:
    """정확 계산 결과."""

    status: Status
    method: str
    ground_state_energy: float | None = None
    ground_state: dict[str, int] | None = None
    runtime: float = 0.0
    penalty_status: Status | None = None
    message: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        base = {
            "exact_status": str(self.status),
            "exact_method": self.method,
            "qubo_ground_state_energy": self.ground_state_energy,
            "exact_runtime": self.runtime,
            "penalty_status": str(self.penalty_status) if self.penalty_status else None,
            "exact_message": self.message,
        }
        base.update(self.fields)
        return base


def exhaustive_ground_state(qubo: MasterQUBO, max_variables: int) -> ExactResult:
    """모든 0/1 조합을 열거해 ground state 를 찾는다.

    Args:
        qubo: 대상 QUBO.
        max_variables: 허용되는 최대 변수 수.

    Returns:
        :class:`ExactResult`. 변수 수 초과 시 ``EXACT_QUBO_SKIPPED_TOO_LARGE``.
    """
    n = qubo.num_variables
    if n > max_variables:
        return ExactResult(
            status=Status.EXACT_QUBO_SKIPPED_TOO_LARGE,
            method="exhaustive",
            message=(
                f"logical variables {n} > threshold {max_variables}. "
                "임의로 축소하지 않고 건너뛴다"
            ),
            fields={"exact_num_variables": n},
        )

    labels = qubo.variable_labels
    best_energy = float("inf")
    best_state: dict[str, int] | None = None
    start = time.perf_counter()
    for combo in itertools.product((0, 1), repeat=n):
        assignment = dict(zip(labels, combo))
        energy = qubo.energy(assignment)
        if energy < best_energy:
            best_energy = energy
            best_state = assignment
    runtime = time.perf_counter() - start

    return ExactResult(
        status=Status.EXACT_QUBO_VALIDATED,
        method="exhaustive",
        ground_state_energy=best_energy,
        ground_state=best_state,
        runtime=runtime,
        fields={"exact_num_variables": n},
    )


def miqp_ground_state(qubo: MasterQUBO, cfg: dict) -> ExactResult:
    """Gurobi MIQP 로 QUBO ground state 를 구한다."""
    n = qubo.num_variables
    val = cfg["validation"]
    license_max = int(val["miqp_max_variables"])
    tractable_max = int(val.get("miqp_tractable_max_variables", license_max))
    time_limit = float(val.get("exact_time_limit", cfg["gurobi"]["time_limit"]))

    if n > license_max:
        return ExactResult(
            status=Status.EXACT_QUBO_SKIPPED_TOO_LARGE,
            method="gurobi_miqp",
            message=(
                f"logical variables {n} > 라이선스 상한 {license_max}. "
                "제한 라이선스는 이차항 모델에서 200 변수까지만 허용한다"
            ),
            fields={"exact_num_variables": n, "exact_skip_reason": "license_limit"},
        )
    if n > tractable_max:
        return ExactResult(
            status=Status.EXACT_QUBO_SKIPPED_TOO_LARGE,
            method="gurobi_miqp",
            message=(
                f"logical variables {n} > 계산 가능 상한 {tractable_max}. "
                "penalized QUBO 는 coefficient dynamic range 가 커서 "
                "이 크기를 넘으면 MIQP 가 time limit 까지 끌려간다. "
                "임의로 축소하지 않고 건너뛴다"
            ),
            fields={"exact_num_variables": n, "exact_skip_reason": "intractable"},
        )
    try:
        start = time.perf_counter()
        params = default_params(cfg)
        params["TimeLimit"] = time_limit  # exact validation 전용 제한
        with gurobi_env(params) as env:
            model = gp.Model("penalized_qubo", env=env)
            variables = {
                name: model.addVar(vtype=GRB.BINARY, name=name) for name in qubo.variable_labels
            }
            expr = gp.QuadExpr(qubo.offset)
            for name, coef in qubo.linear.items():
                expr.add(float(coef) * variables[name])
            for (a, b), coef in qubo.quadratic.items():
                expr.add(float(coef) * variables[a] * variables[b])
            model.setObjective(expr, GRB.MINIMIZE)
            model.optimize()
            status = map_model_status(model.Status)
            if status is not Status.OPTIMAL:
                # time limit 으로 얻은 incumbent 를 ground state 라고 부르지 않는다.
                fields = {
                    "exact_num_variables": n,
                    "exact_time_limit": time_limit,
                    "exact_incumbent_energy": (
                        float(model.ObjVal) if model.SolCount > 0 else None
                    ),
                    "exact_mip_gap": (
                        float(model.MIPGap) if model.SolCount > 0 else None
                    ),
                }
                return ExactResult(
                    status=status,
                    method="gurobi_miqp",
                    message=(
                        f"MIQP status={int(model.Status)}. "
                        f"time limit {time_limit}s 내에 ground state 를 증명하지 못했다"
                    ),
                    fields=fields,
                )
            state = {name: int(round(var.X)) for name, var in variables.items()}
            energy = float(model.ObjVal)
        runtime = time.perf_counter() - start
    except Exception as exc:
        status, message = classify_gurobi_error(exc)
        return ExactResult(
            status=status, method="gurobi_miqp", message=message,
            fields={"exact_num_variables": n},
        )

    return ExactResult(
        status=Status.EXACT_QUBO_VALIDATED,
        method="gurobi_miqp",
        ground_state_energy=energy,
        ground_state=state,
        runtime=runtime,
        fields={"exact_num_variables": n, "exact_time_limit": time_limit},
    )


def check_penalty_sufficiency(
    result: ExactResult,
    qubo: MasterQUBO,
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    cfg: dict,
) -> ExactResult:
    """Ground state 가 MP-feasible 한지 확인해 penalty 충분성을 판정한다.

    Ground state 가 MP-infeasible 하면 penalty 가 부족하다는 뜻이다.
    이때 penalty 를 자동으로 바꾸지 않고 ``PENALTY_INSUFFICIENT`` 로 기록한다.
    """
    if result.ground_state is None:
        result.penalty_status = None
        return result

    decoded = decode_sample(
        qubo, inst, cuts, result.ground_state, result.ground_state_energy or 0.0, 0, cfg
    )
    result.penalty_status = (
        Status.SUCCESS if decoded.mp_feasible else Status.PENALTY_INSUFFICIENT
    )
    result.fields.update(
        {
            "ground_state_mp_feasible": decoded.mp_feasible,
            "ground_state_max_mp_violation": decoded.max_mp_violation,
            "ground_state_decoded_theta": decoded.decoded_theta,
            "ground_state_decoded_y": decoded.decoded_y.astype(int).tolist(),
        }
    )
    return result


def validate_qubo_exactly(
    qubo: MasterQUBO,
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    cfg: dict,
) -> ExactResult:
    """가능한 방법으로 QUBO ground state 를 구하고 penalty 충분성을 검증한다.

    exhaustive enumeration 을 우선 시도하고, 크기 초과 시 Gurobi MIQP 를 쓴다.
    둘 다 불가능하면 ``EXACT_QUBO_SKIPPED_TOO_LARGE`` 를 반환한다.
    """
    result = exhaustive_ground_state(
        qubo, int(cfg["validation"]["exhaustive_max_variables"])
    )
    if result.status is Status.EXACT_QUBO_SKIPPED_TOO_LARGE:
        result = miqp_ground_state(qubo, cfg)
    return check_penalty_sufficiency(result, qubo, inst, cuts, cfg)
