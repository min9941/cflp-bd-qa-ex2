"""Benders lower/upper bound certificate.

원칙(명세 6절):

- **certified lower bound** 는 Gurobi 가 푼 continuous MP 의 최적값
  또는 유효한 ``ObjBound`` 에서만 온다.
- QA/SA 의 목적값은 절대 lower bound 로 쓰지 않는다.
- **upper bound** 는 SP 로 평가한 feasible y 에서 온다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..status import Status


@dataclass
class BoundState:
    """현재까지의 lower/upper bound 상태."""

    certified_lower_bound: float = float("-inf")
    best_upper_bound: float = float("inf")
    best_y: list[int] | None = None
    lower_bound_source: str = "none"
    lower_bound_status: str = str(Status.SUCCESS)
    history: list[dict[str, Any]] = field(default_factory=list)

    def update_lower(self, value: float, source: str, status: Status) -> None:
        """Lower bound 를 갱신한다 (단조 증가만 허용).

        Args:
            value: 새 lower bound 후보.
            source: 출처 설명 (예: ``"continuous_mp_optimal"``).
            status: 해당 solve 의 status.
        """
        if value > self.certified_lower_bound:
            self.certified_lower_bound = float(value)
            self.lower_bound_source = source
            self.lower_bound_status = str(status)

    def update_upper(self, value: float, y: list[int]) -> bool:
        """Upper bound 를 갱신한다.

        Returns:
            갱신되었으면 True.
        """
        if value < self.best_upper_bound:
            self.best_upper_bound = float(value)
            self.best_y = list(y)
            return True
        return False

    def relative_gap(self) -> float:
        """``(UB - LB) / max(|UB|, 1e-12)`` 를 반환한다."""
        ub, lb = self.best_upper_bound, self.certified_lower_bound
        if ub == float("inf") or lb == float("-inf"):
            return float("inf")
        return (ub - lb) / max(abs(ub), 1e-12)

    def check_bound_order(self, tolerance: float) -> str | None:
        """``UB < LB`` 가 허용 오차를 넘게 발생했는지 확인한다.

        Returns:
            문제가 있으면 설명 문자열, 없으면 ``None``.
        """
        ub, lb = self.best_upper_bound, self.certified_lower_bound
        if ub == float("inf") or lb == float("-inf"):
            return None
        scale = max(1.0, abs(ub))
        if (lb - ub) / scale > tolerance:
            return (
                f"UB < LB 위반: UB={ub!r}, LB={lb!r}. "
                "수치 또는 구현 오류로 간주한다"
            )
        return None
