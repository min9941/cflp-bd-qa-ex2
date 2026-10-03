"""Theta(Benders recourse 변수) encoding.

파일럿 baseline 은 bound 전략을 비교하지 않는다(명세 8절).

    theta_L = 0
    theta_U = U_obj = sum_j f_j + sum_i sum_j c_ij d_i
    delta_theta = (theta_U - theta_L) / (2^b - 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pure_exponential import EncodingError, PureExponentialEncoding


@dataclass(frozen=True)
class ThetaEncoding:
    """theta encoding 과 그 진단값."""

    encoding: PureExponentialEncoding
    theta_bits: int
    theta_lower_bound: float
    theta_upper_bound: float
    theta_range: float
    theta_delta: float
    theta_relative_resolution: float
    theta_encoded_max: float

    def decode(self, bits: dict[str, int] | list[int]) -> float:
        """bit 로부터 theta 값을 복원한다."""
        return self.encoding.decode(bits)

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict 를 반환한다(명세 8절 기록 항목)."""
        return {
            "theta_bits": self.theta_bits,
            "theta_lower_bound": self.theta_lower_bound,
            "theta_upper_bound": self.theta_upper_bound,
            "theta_range": self.theta_range,
            "theta_delta": self.theta_delta,
            "theta_relative_resolution": self.theta_relative_resolution,
            "theta_encoded_max": self.theta_encoded_max,
        }


def build_theta_encoding(objective_upper_bound: float, theta_bits: int) -> ThetaEncoding:
    """theta encoding 을 구성한다.

    Args:
        objective_upper_bound: U_obj. theta 의 상한으로 사용한다.
        theta_bits: 사용할 bit 수 b_theta.

    Returns:
        구성된 :class:`ThetaEncoding`.

    Raises:
        EncodingError: 상한이 0 이하이거나 bit 수가 1 미만인 경우
            (상태 ``THETA_RANGE_INVALID`` 에 해당).
    """
    if theta_bits < 1:
        raise EncodingError("theta_bits 는 1 이상이어야 한다 (THETA_RANGE_INVALID)")
    theta_lower = 0.0
    theta_upper = float(objective_upper_bound)
    if theta_upper <= theta_lower:
        raise EncodingError(
            f"theta 범위가 유효하지 않다: [{theta_lower}, {theta_upper}] (THETA_RANGE_INVALID)"
        )

    theta_range = theta_upper - theta_lower
    delta = theta_range / (2 ** theta_bits - 1)
    enc = PureExponentialEncoding(n_bits=theta_bits, delta=delta, offset=theta_lower, label="theta")

    return ThetaEncoding(
        encoding=enc,
        theta_bits=theta_bits,
        theta_lower_bound=theta_lower,
        theta_upper_bound=theta_upper,
        theta_range=theta_range,
        theta_delta=delta,
        theta_relative_resolution=delta / theta_range,
        theta_encoded_max=enc.max_value,
    )
