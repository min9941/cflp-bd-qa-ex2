"""QUBO sample 디코딩과 MP feasibility 판정.

sample 선택 규칙(명세 19절):

1. unembedded logical sample 을 QUBO energy 오름차순으로 정렬
2. y 와 theta 를 original scale 로 decode
3. 현재 MP 의 capacity constraint 와 **모든 활성 optimality cut** 을 검사
4. MP-feasible 한 첫 sample 을 선택
5. 없으면 ``NO_FEASIBLE_SAMPLE``

MP feasibility 는 auxiliary slack equality 의 exact residual 이 아니라
**원래 MP inequality** 를 기준으로 한다.

시설을 추가하거나 제거하는 repair 는 금지한다.
``majority_vote`` 외의 시설-level repair 또는 local search 를 적용하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..benders.cuts import OptimalityCut
from ..data.generator import Instance
from ..models.master import mp_objective, mp_violations
from ..status import Status
from .builder import MasterQUBO, ROLE_Y


@dataclass
class DecodedSample:
    """디코딩된 sample 하나."""

    decoded_y: np.ndarray
    decoded_theta: float
    energy: float
    decoded_mp_objective: float
    capacity_violation: float
    max_cut_violation: float
    max_mp_violation: float
    normalized_max_mp_violation: float
    penalty_residual: float
    mp_feasible: bool
    borderline_feasible: bool
    sample_rank: int
    num_occurrences: int = 1
    chain_break_fraction: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        return {
            "decoded_y": self.decoded_y.astype(int).tolist(),
            "decoded_theta": self.decoded_theta,
            "energy": self.energy,
            "decoded_mp_objective": self.decoded_mp_objective,
            "capacity_violation": self.capacity_violation,
            "max_cut_violation": self.max_cut_violation,
            "max_mp_violation": self.max_mp_violation,
            "normalized_max_mp_violation": self.normalized_max_mp_violation,
            "penalty_residual": self.penalty_residual,
            "mp_feasible": self.mp_feasible,
            "borderline_feasible": self.borderline_feasible,
            "sample_rank": self.sample_rank,
            "num_occurrences": self.num_occurrences,
            "chain_break_fraction": self.chain_break_fraction,
        }


def decode_sample(
    qubo: MasterQUBO,
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    assignment: dict[str, int],
    energy: float,
    rank: int,
    cfg: dict,
    num_occurrences: int = 1,
    chain_break_fraction: float | None = None,
) -> DecodedSample:
    """단일 sample 을 original scale 로 디코딩하고 feasibility 를 판정한다."""
    y_names = qubo.variables_by_role(ROLE_Y)
    decoded_y = np.array([int(assignment[name]) for name in y_names], dtype=int)
    decoded_theta = qubo.theta_encoding.decode(
        {name: int(assignment[name]) for name in qubo.theta_encoding.encoding.variable_names()}
    )

    violations = mp_violations(inst, decoded_y, decoded_theta, cuts)
    tol = float(cfg["validation"]["mp_feasibility_tolerance"])
    borderline_tol = float(cfg["validation"].get("borderline_feasibility_tolerance", 1e-4))

    scale = max(1.0, abs(float(inst.objective_upper_bound)))
    max_violation = violations["max_mp_violation"]

    # penalty residual: 모든 equality residual 의 최대 절댓값.
    # dual 계수가 실수이므로 MP-feasible 이어도 0이 아닐 수 있다.
    penalty_residual = 0.0
    for form in qubo.residual_forms.values():
        penalty_residual = max(penalty_residual, abs(form.evaluate(assignment)))

    return DecodedSample(
        decoded_y=decoded_y,
        decoded_theta=decoded_theta,
        energy=float(energy),
        decoded_mp_objective=mp_objective(inst, decoded_y, decoded_theta),
        capacity_violation=violations["capacity_violation"],
        max_cut_violation=violations["max_cut_violation"],
        max_mp_violation=max_violation,
        normalized_max_mp_violation=max_violation / scale,
        penalty_residual=penalty_residual,
        mp_feasible=max_violation <= tol,
        borderline_feasible=tol < max_violation <= borderline_tol,
        sample_rank=rank,
        num_occurrences=num_occurrences,
        chain_break_fraction=chain_break_fraction,
    )


def select_feasible_sample(
    qubo: MasterQUBO,
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    samples: Iterable[tuple[dict[str, int], float, int, float | None]],
    cfg: dict,
) -> tuple[DecodedSample | None, list[DecodedSample], Status]:
    """energy 오름차순으로 검사하여 첫 MP-feasible sample 을 고른다.

    Args:
        qubo: 대상 QUBO.
        inst: 대상 인스턴스.
        cuts: 활성 cut 목록.
        samples: ``(assignment, energy, num_occurrences, chain_break_fraction)``
            튜플의 iterable. **energy 오름차순으로 정렬되어 있어야 한다.**
        cfg: 전체 설정 dict.

    Returns:
        ``(selected, all_decoded, status)``. feasible sample 이 없으면
        ``selected`` 는 ``None`` 이고 status 는 ``NO_FEASIBLE_SAMPLE`` 이다.
    """
    decoded_all: list[DecodedSample] = []
    selected: DecodedSample | None = None

    for rank, (assignment, energy, occurrences, cbf) in enumerate(samples):
        decoded = decode_sample(
            qubo, inst, cuts, assignment, energy, rank, cfg, occurrences, cbf
        )
        decoded_all.append(decoded)
        if selected is None and decoded.mp_feasible:
            selected = decoded

    if selected is None:
        return None, decoded_all, Status.NO_FEASIBLE_SAMPLE
    return selected, decoded_all, Status.SUCCESS


def feasible_sample_rate(decoded: Sequence[DecodedSample]) -> float:
    """MP-feasible sample 의 비율(occurrence 가중)을 계산한다."""
    total = sum(d.num_occurrences for d in decoded)
    if total == 0:
        return 0.0
    feasible = sum(d.num_occurrences for d in decoded if d.mp_feasible)
    return feasible / total
