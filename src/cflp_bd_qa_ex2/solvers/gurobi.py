"""Gurobi 환경 관리와 오류 분류.

제한(size-limited) 라이선스 환경에서도 프로젝트 전체가 crash 하지 않도록
라이선스/크기 오류를 status 로 분류해 돌려준다.

제한 라이선스의 알려진 한계:

- 선형 모델: 변수 2000개, 선형 제약 2000개
- 이차항이 있는 모델(QP/MIQP 등): 변수 200개

따라서 32x100 인스턴스는 subproblem 의 x_ij 가 3200개이므로
제한 라이선스에서는 SP 부터 실패한다.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..status import Status

try:  # pragma: no cover - 환경 의존
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
    GUROBI_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - 환경 의존
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]
    GUROBI_AVAILABLE = False
    GUROBI_IMPORT_ERROR = str(exc)


SIZE_LIMIT_ERRNO = 10010
NO_LICENSE_ERRNOS = (10009, 10002)


class GurobiUnavailable(RuntimeError):
    """Gurobi 를 사용할 수 없을 때 발생하는 예외."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class SolveOutcome:
    """Gurobi solve 하나의 결과 요약."""

    status: Status
    objective: float | None = None
    bound: float | None = None
    runtime: float = 0.0
    gurobi_status: int | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """정상적으로 최적해를 얻었는지 여부."""
        return self.status is Status.OPTIMAL


def classify_gurobi_error(exc: Exception) -> tuple[Status, str]:
    """Gurobi 예외를 프로젝트 status 로 분류한다.

    Returns:
        ``(status, message)`` 튜플.
    """
    errno = getattr(exc, "errno", None)
    text = str(exc)
    if errno == SIZE_LIMIT_ERRNO or "size-limited" in text:
        return Status.GUROBI_SIZE_LIMIT, text
    if errno in NO_LICENSE_ERRNOS or "license" in text.lower():
        return Status.GUROBI_LICENSE_UNAVAILABLE, text
    return Status.SOLVER_FAILED, text


def map_model_status(model_status: int) -> Status:
    """Gurobi 모델 status 코드를 프로젝트 status 로 매핑한다.

    해석할 수 없는 코드는 ``SOLVER_FAILED`` 로 돌린다.
    안전하게 해석할 수 없는 status 를 임의로 성공 처리하지 않는다.

    ``INF_OR_UNBD`` 는 ``INFEASIBLE`` 로 단정하지 않는다. Gurobi 는
    ``DualReductions=0`` 으로 재최적화해야 둘을 구분할 수 있으므로
    별도 상태 ``INFEASIBLE_OR_UNBOUNDED`` 를 돌려준다
    (:func:`disambiguate_inf_or_unbd` 참고).
    """
    mapping = {
        GRB.OPTIMAL: Status.OPTIMAL,
        GRB.INFEASIBLE: Status.INFEASIBLE,
        GRB.INF_OR_UNBD: Status.INFEASIBLE_OR_UNBOUNDED,
        GRB.UNBOUNDED: Status.UNBOUNDED,
        GRB.TIME_LIMIT: Status.TIME_LIMIT,
    }
    return mapping.get(model_status, Status.SOLVER_FAILED)


def check_gurobi_available() -> tuple[bool, Status | None, str]:
    """Gurobi 사용 가능 여부를 preflight 로 확인한다.

    import 가능 여부와 환경/모델 생성 가능 여부를 모두 확인한다.

    Returns:
        ``(available, status_or_None, message)``.
    """
    if not GUROBI_AVAILABLE:
        return False, Status.GUROBI_LICENSE_UNAVAILABLE, f"gurobipy import 실패: {GUROBI_IMPORT_ERROR}"
    try:
        with gurobi_env({"OutputFlag": 0}) as env:
            model = gp.Model(env=env)
            model.addVar()
            model.optimize()
        return True, None, f"gurobi {gp.gurobi.version()}"
    except Exception as exc:  # pragma: no cover - 환경 의존
        status, message = classify_gurobi_error(exc)
        return False, status, message


def probe_size_limits() -> dict[str, Any]:
    """현재 라이선스의 선형/이차 모델 크기 한계를 실제로 탐지한다.

    Returns:
        ``{"linear_limit": int|None, "quadratic_limit": int|None, "restricted": bool}``.
        ``None`` 은 해당 한계를 확인하지 못했다는 뜻이다.
    """
    if not GUROBI_AVAILABLE:
        return {"linear_limit": None, "quadratic_limit": None, "restricted": None}

    def _fits(n: int, quadratic: bool) -> bool:
        try:
            with gurobi_env({"OutputFlag": 0}) as env:
                model = gp.Model(env=env)
                x = model.addVars(n, vtype=GRB.BINARY)
                if quadratic:
                    model.setObjective(gp.quicksum(x[i] * x[(i + 1) % n] for i in range(n)))
                else:
                    model.setObjective(gp.quicksum(x[i] for i in range(n)))
                model.addConstr(gp.quicksum(x[i] for i in range(n)) >= 1)
                model.optimize()
            return True
        except Exception:
            return False

    linear_limit = None if _fits(2001, False) else 2000
    quadratic_limit = None if _fits(201, True) else 200
    return {
        "linear_limit": linear_limit,
        "quadratic_limit": quadratic_limit,
        "restricted": linear_limit is not None or quadratic_limit is not None,
    }


def disambiguate_inf_or_unbd(model: Any) -> Status:
    """``INF_OR_UNBD`` 를 ``DualReductions=0`` 재최적화로 구분한다.

    Args:
        model: 이미 ``INF_OR_UNBD`` 로 끝난 Gurobi 모델.

    Returns:
        ``INFEASIBLE``, ``UNBOUNDED``, 또는 여전히 구분되지 않으면
        ``INFEASIBLE_OR_UNBOUNDED``.
    """
    try:
        model.setParam("DualReductions", 0)
        model.optimize()
    except Exception:
        return Status.INFEASIBLE_OR_UNBOUNDED
    return map_model_status(model.Status)


@contextmanager
def gurobi_env(params: dict[str, Any] | None = None) -> Iterator[Any]:
    """Gurobi 환경을 생성하고 반드시 해제하는 context manager."""
    if not GUROBI_AVAILABLE:
        raise GurobiUnavailable(
            Status.GUROBI_LICENSE_UNAVAILABLE, f"gurobipy import 실패: {GUROBI_IMPORT_ERROR}"
        )
    env = gp.Env(params=dict(params or {}))
    try:
        yield env
    finally:
        env.dispose()


def default_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """설정으로부터 Gurobi 파라미터 dict 를 만든다."""
    gcfg = cfg["gurobi"]
    return {
        "OutputFlag": int(gcfg.get("output_flag", 0)),
        "TimeLimit": float(gcfg["time_limit"]),
        "FeasibilityTol": float(gcfg["feasibility_tolerance"]),
    }


@contextmanager
def timed() -> Iterator[list[float]]:
    """경과 시간을 리스트의 첫 원소에 기록하는 context manager."""
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
