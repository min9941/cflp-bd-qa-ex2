"""Constraint penalty coefficient 관리.

수학 표기와 코드에서 constraint penalty coefficient 는 항상 ``penalty_alpha``
라고 부른다. 일반 이름 ``alpha`` 만 단독으로 쓰지 않는다.

이것은 embedding 의 **chain strength 와 전혀 다른 개념**이다.
chain strength 는 같은 logical variable 을 나타내는 physical qubit 들을
묶는 힘이고, ``penalty_alpha`` 는 원래 문제의 제약 위반에 부과하는 비용이다.

파일럿에서는 모든 constraint 에 동일한 규칙을 쓴다.

    penalty_alpha_capacity = margin * U_obj
    penalty_alpha_cut_k    = margin * U_obj      for all k

다음은 구현하지 않는다: constraint 별 tuning, adaptive update, penalty sweep,
Ocean 의 ``10 x max objective bias`` 자동 규칙, constraint normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PenaltySpec:
    """QUBO 하나에 적용된 penalty 계수 모음."""

    penalty_alpha_capacity: float
    penalty_alpha_cuts: tuple[float, ...]
    margin: float
    objective_upper_bound: float
    per_constraint: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        cuts = list(self.penalty_alpha_cuts)
        return {
            "penalty_alpha_capacity": self.penalty_alpha_capacity,
            "penalty_alpha_cut_min": min(cuts) if cuts else None,
            "penalty_alpha_cut_max": max(cuts) if cuts else None,
            "penalty_margin": self.margin,
            "penalty_per_constraint": self.per_constraint,
            "penalty_objective_upper_bound": self.objective_upper_bound,
        }


def build_penalty_spec(
    objective_upper_bound: float,
    num_cuts: int,
    cfg: dict,
) -> PenaltySpec:
    """파일럿 규칙에 따라 penalty 계수를 만든다.

    Args:
        objective_upper_bound: U_obj.
        num_cuts: 현재 활성 cut 수 K.
        cfg: 전체 설정 dict.

    Returns:
        :class:`PenaltySpec`. 각 constraint 에 개별 값을 전달하되
        파일럿에서는 모두 ``margin * U_obj`` 로 동일하다.
    """
    margin = float(cfg["penalty"]["margin"])
    value = margin * float(objective_upper_bound)
    return PenaltySpec(
        penalty_alpha_capacity=value,
        penalty_alpha_cuts=tuple(value for _ in range(num_cuts)),
        margin=margin,
        objective_upper_bound=float(objective_upper_bound),
        per_constraint=bool(cfg["penalty"].get("per_constraint", True)),
    )
