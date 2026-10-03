"""Capacity slack 과 optimality-cut slack encoding.

두 slack 은 성격이 다르다.

- **capacity slack**: ``sum_j s_j y_j - D - q_cap = 0``.
  s_j 와 D 가 모두 정수이므로 ``delta = 1`` 의 pure exponential 로
  **정확히** 표현된다. 근사하지 않는다.

- **cut slack**: ``theta + sum_j s_j v_j^k y_j - sum_i u_i^k - q_k = 0``.
  dual 계수 v, u 가 실수이므로 격자로 정확히 표현되지 않는다.
  따라서 MP-feasible sample 이 penalty residual 0 을 갖는다고 가정하면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pure_exponential import EncodingError, PureExponentialEncoding, bits_for_range


class CutSlackRangeInvalid(ValueError):
    """Cut slack 범위가 허용 오차를 넘어 음수인 경우 (``CUT_SLACK_RANGE_INVALID``)."""


@dataclass(frozen=True)
class CapacitySlackEncoding:
    """Capacity slack encoding."""

    encoding: PureExponentialEncoding
    capacity_slack_bits: int
    capacity_slack_max: int

    def decode(self, bits: dict[str, int] | list[int]) -> float:
        """bit 로부터 capacity slack 값을 복원한다."""
        return self.encoding.decode(bits)


@dataclass(frozen=True)
class CutSlackEncoding:
    """단일 optimality cut 의 slack encoding."""

    encoding: PureExponentialEncoding
    cut_index: int
    cut_slack_bits: int
    cut_slack_max: float
    cut_slack_delta: float
    clamped_from_negative: bool = False

    def decode(self, bits: dict[str, int] | list[int]) -> float:
        """bit 로부터 cut slack 값을 복원한다."""
        return self.encoding.decode(bits)


def build_capacity_slack_encoding(
    total_capacity: int, total_demand: int, delta: float = 1.0
) -> CapacitySlackEncoding:
    """Capacity slack encoding 을 구성한다.

    Args:
        total_capacity: sum_j s_j.
        total_demand: D = sum_i d_i.
        delta: 격자 간격. 명세상 반드시 1 이다.

    Returns:
        구성된 :class:`CapacitySlackEncoding`.

    Raises:
        EncodingError: ``delta != 1`` 이거나 총용량이 총수요보다 작은 경우.
    """
    if delta != 1.0:
        raise EncodingError("capacity slack 의 delta 는 1이어야 한다 (근사 금지)")
    slack_max = int(total_capacity) - int(total_demand)
    if slack_max < 0:
        raise EncodingError(
            f"총용량이 총수요보다 작다: sum(s_j)={total_capacity} < D={total_demand}. "
            "이 인스턴스는 MP 자체가 infeasible 하다"
        )
    n_bits = bits_for_range(float(slack_max), delta)
    enc = PureExponentialEncoding(n_bits=n_bits, delta=delta, offset=0.0, label="qcap")
    return CapacitySlackEncoding(
        encoding=enc, capacity_slack_bits=n_bits, capacity_slack_max=slack_max
    )


def cut_slack_max(
    theta_upper_bound: float,
    capacity: np.ndarray,
    dual_v: np.ndarray,
    dual_u_sum: float,
) -> float:
    """Cut k 의 slack 상한 S_k_max 를 계산한다.

    ``S_k_max = theta_U + sum_j s_j v_j^k - sum_i u_i^k``.

    이는 residual ``theta + sum_j s_j v_j y_j - sum_i u_i`` 를
    ``theta = theta_U``, ``y = 1`` 에서 평가한 값이며 (v_j >= 0 이므로) 최대값이다.
    """
    return float(theta_upper_bound + float(capacity @ dual_v) - float(dual_u_sum))


def build_cut_slack_encoding(
    cut_index: int,
    theta_upper_bound: float,
    capacity: np.ndarray,
    dual_v: np.ndarray,
    dual_u_sum: float,
    theta_delta: float,
    delta_ratio_to_theta: float,
    negative_tolerance: float = 1e-6,
) -> CutSlackEncoding:
    """Optimality cut 의 slack encoding 을 구성한다.

    bit 수는 자동 계산하며 임의로 cap 하지 않는다.

    Args:
        cut_index: cut 번호 k.
        theta_upper_bound: theta_U.
        capacity: s_j 배열.
        dual_v: v_j^k 배열.
        dual_u_sum: sum_i u_i^k.
        theta_delta: delta_theta.
        delta_ratio_to_theta: r_cut. delta_cut = r_cut * delta_theta.
        negative_tolerance: S_k_max 가 이 값만큼 음수이면 0으로 처리한다.

    Returns:
        구성된 :class:`CutSlackEncoding`. ``S_k_max = 0`` 이면 bit 수는 0이다.

    Raises:
        CutSlackRangeInvalid: ``S_k_max < -negative_tolerance`` 인 경우.
    """
    delta_cut = delta_ratio_to_theta * theta_delta
    raw_max = cut_slack_max(theta_upper_bound, capacity, dual_v, dual_u_sum)

    clamped = False
    if raw_max < -negative_tolerance:
        raise CutSlackRangeInvalid(
            f"cut {cut_index}: S_k_max = {raw_max!r} < -{negative_tolerance} "
            "(CUT_SLACK_RANGE_INVALID)"
        )
    if raw_max < 0.0:
        clamped = True
        raw_max = 0.0

    n_bits = bits_for_range(raw_max, delta_cut)
    enc = PureExponentialEncoding(
        n_bits=n_bits, delta=delta_cut, offset=0.0, label=f"qcut{cut_index}"
    )
    return CutSlackEncoding(
        encoding=enc,
        cut_index=cut_index,
        cut_slack_bits=n_bits,
        cut_slack_max=raw_max,
        cut_slack_delta=delta_cut,
        clamped_from_negative=clamped,
    )


def summarize_cut_slack_bits(encodings: list[CutSlackEncoding]) -> dict[str, Any]:
    """Cut slack bit 수 통계를 반환한다 (명세 17절 기록 항목)."""
    bits = [e.cut_slack_bits for e in encodings]
    if not bits:
        return {
            "cut_slack_bits_total": 0,
            "cut_slack_bits_min": None,
            "cut_slack_bits_mean": None,
            "cut_slack_bits_max": None,
        }
    return {
        "cut_slack_bits_total": int(sum(bits)),
        "cut_slack_bits_min": int(min(bits)),
        "cut_slack_bits_mean": float(np.mean(bits)),
        "cut_slack_bits_max": int(max(bits)),
    }
