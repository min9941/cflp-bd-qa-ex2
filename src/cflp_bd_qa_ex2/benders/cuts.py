"""Benders optimality cut 표현과 유효성 검사.

Cut k 의 형태:

    theta >= sum_i u_i^k - sum_j s_j v_j^k y_j

동치 형태:

    theta + sum_j s_j v_j^k y_j >= sum_i u_i^k

어떤 dual-feasible (u, v) 에 대해서도 dual objective 는 모든 y 에서 Q(y) 의
하한이므로(weak duality) cut 은 valid 하다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptimalityCut:
    """하나의 Benders optimality cut.

    Attributes:
        index: cut 번호 k (0부터).
        u: u^k, shape ``(n_customers,)``.
        v: v^k, shape ``(n_facilities,)``.
        rhs_constant: sum_i u_i^k.
        y_coefficients: s_j v_j^k, shape ``(n_facilities,)``.
        generated_at_iteration: 이 cut 이 생성된 Benders iteration.
        dual_minimality_status: 생성 시점의 guard 진단 결과.
    """

    index: int
    u: np.ndarray
    v: np.ndarray
    rhs_constant: float
    y_coefficients: np.ndarray
    generated_at_iteration: int
    dual_minimality_status: str = "NOT_CHECKED"

    def residual(self, y: np.ndarray, theta: float) -> float:
        """``theta + sum_j s_j v_j y_j - sum_i u_i`` 를 계산한다.

        0 이상이면 cut 을 만족한다.
        """
        return float(theta + self.y_coefficients @ np.asarray(y, dtype=float) - self.rhs_constant)

    def violation(self, y: np.ndarray, theta: float) -> float:
        """Cut 위반량 ``max(0, -residual)`` 을 반환한다."""
        return max(0.0, -self.residual(y, theta))

    def lower_bound_at(self, y: np.ndarray) -> float:
        """주어진 y 에서 이 cut 이 강제하는 theta 하한."""
        return float(self.rhs_constant - self.y_coefficients @ np.asarray(y, dtype=float))

    @property
    def signature(self) -> str:
        """중복 cut 판정을 위한 결정적 지문."""
        blob = np.concatenate(
            [np.round(self.y_coefficients, 9), np.round([self.rhs_constant], 9)]
        ).tobytes()
        return hashlib.sha256(blob).hexdigest()[:32]

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        return {
            "cut_index": self.index,
            "generated_at_iteration": self.generated_at_iteration,
            "cut_rhs_constant": self.rhs_constant,
            "cut_y_coefficients": self.y_coefficients.tolist(),
            "cut_dual_u": self.u.tolist(),
            "cut_dual_v": self.v.tolist(),
            "cut_signature": self.signature,
            "dual_minimality_status": self.dual_minimality_status,
        }


def build_optimality_cut(
    index: int,
    capacity: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    iteration: int,
    dual_minimality_status: str = "NOT_CHECKED",
) -> OptimalityCut:
    """Dual 해로부터 optimality cut 을 만든다.

    Args:
        index: cut 번호.
        capacity: s_j.
        u: dual u^k.
        v: dual v^k.
        iteration: 생성된 Benders iteration.
        dual_minimality_status: guard 진단 결과 문자열.

    Returns:
        생성된 :class:`OptimalityCut`.
    """
    return OptimalityCut(
        index=index,
        u=np.asarray(u, dtype=float).copy(),
        v=np.asarray(v, dtype=float).copy(),
        rhs_constant=float(np.asarray(u, dtype=float).sum()),
        y_coefficients=np.asarray(capacity, dtype=float) * np.asarray(v, dtype=float),
        generated_at_iteration=iteration,
        dual_minimality_status=dual_minimality_status,
    )


def cut_is_valid_at(cut: OptimalityCut, y: np.ndarray, true_recourse: float, tol: float = 1e-6) -> bool:
    """Cut 이 주어진 y 에서 valid 한지 확인한다.

    ``cut.lower_bound_at(y) <= Q(y) + tol`` 이어야 한다.
    """
    return cut.lower_bound_at(y) <= true_recourse + tol
