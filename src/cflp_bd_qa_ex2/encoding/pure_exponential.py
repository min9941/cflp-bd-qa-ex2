"""Pure exponential(순수 지수) binary encoding.

마지막 bit 를 줄이는 truncated encoding 은 사용하지 않는다(명세 8절, 10절).

인코딩 형태는 다음과 같다.

    value = offset + delta * sum_{p=0}^{b-1} 2^p * z_p

따라서 표현 가능한 값은 ``offset`` 부터 ``offset + delta * (2^b - 1)`` 까지의
등간격 격자이며, 격자 간격은 ``delta`` 이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class EncodingError(ValueError):
    """Encoding 정의가 유효하지 않을 때 발생하는 예외."""


@dataclass(frozen=True)
class PureExponentialEncoding:
    """순수 지수 binary encoding 정의.

    Attributes:
        n_bits: 사용하는 bit 수 b. 0이면 상수 ``offset`` 만 표현한다.
        delta: 격자 간격.
        offset: 하한(= 모든 bit 가 0일 때의 값).
        label: 변수 label prefix (예: ``"theta"``).
    """

    n_bits: int
    delta: float
    offset: float = 0.0
    label: str = "z"

    def __post_init__(self) -> None:
        if self.n_bits < 0:
            raise EncodingError("n_bits 는 0 이상이어야 한다")
        if self.n_bits > 0 and self.delta <= 0:
            raise EncodingError("delta 는 양수여야 한다")

    @property
    def max_value(self) -> float:
        """표현 가능한 최대값 offset + delta * (2^b - 1)."""
        return self.offset + self.delta * (2 ** self.n_bits - 1)

    @property
    def span(self) -> float:
        """offset 으로부터의 표현 범위 폭."""
        return self.delta * (2 ** self.n_bits - 1)

    def variable_names(self) -> list[str]:
        """bit 변수 이름 목록을 낮은 자리부터 반환한다."""
        return [f"{self.label}_{p}" for p in range(self.n_bits)]

    def coefficients(self) -> list[float]:
        """각 bit 변수의 계수 delta * 2^p 목록."""
        return [self.delta * (2 ** p) for p in range(self.n_bits)]

    def decode(self, bits: dict[str, int] | list[int]) -> float:
        """bit 값으로부터 원래 scale 의 값을 복원한다.

        Args:
            bits: 변수명 -> 0/1 dict, 또는 낮은 자리부터의 0/1 리스트.

        Returns:
            디코딩된 실수 값.
        """
        if isinstance(bits, dict):
            values = [int(bits[name]) for name in self.variable_names()]
        else:
            values = [int(b) for b in bits]
            if len(values) != self.n_bits:
                raise EncodingError(f"bit 개수 불일치: {len(values)} != {self.n_bits}")
        total = sum(v * (2 ** p) for p, v in enumerate(values))
        return self.offset + self.delta * total

    def encode_nearest(self, value: float) -> list[int]:
        """값을 표현 가능한 가장 가까운 격자점의 bit 목록으로 변환한다.

        범위를 벗어나면 clip 한다. 반환은 낮은 자리부터의 0/1 리스트이다.
        """
        if self.n_bits == 0:
            return []
        steps = round((value - self.offset) / self.delta)
        steps = max(0, min(steps, 2 ** self.n_bits - 1))
        return [(int(steps) >> p) & 1 for p in range(self.n_bits)]


def bits_for_range(max_value: float, delta: float) -> int:
    """``[0, max_value]`` 를 간격 ``delta`` 로 덮는 데 필요한 bit 수.

    b = ceil(log2(max_value / delta + 1)) 이다. ``max_value`` 가 0이면 0을 반환한다.

    Args:
        max_value: 덮어야 하는 최대값(0 이상).
        delta: 격자 간격(양수).

    Returns:
        필요한 bit 수.

    Raises:
        EncodingError: ``max_value`` 가 음수이거나 ``delta`` 가 0 이하인 경우.
    """
    if delta <= 0:
        raise EncodingError("delta 는 양수여야 한다")
    if max_value < 0:
        raise EncodingError(f"max_value 는 음수일 수 없다: {max_value}")
    if max_value == 0:
        return 0
    return int(math.ceil(math.log2(max_value / delta + 1.0)))
