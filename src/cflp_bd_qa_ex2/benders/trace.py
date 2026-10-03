"""Benders iteration 기록 구조.

MP-level 결과와 Benders-level 결과는 서로 다른 ``record_type`` 으로 구분한다
(명세 21절).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RECORD_TYPE_ITERATION = "benders_iteration"
RECORD_TYPE_RUN = "benders_run"


@dataclass
class IterationRecord:
    """Benders iteration 하나의 기록 (``record_type = benders_iteration``)."""

    instance_id: str
    instance_name: str
    instance_seed: int
    trajectory_source: str
    benders_iteration: int
    num_benders_cuts: int
    fields: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """평탄화된 결과 dict 를 반환한다."""
        base = {
            "record_type": RECORD_TYPE_ITERATION,
            "instance_id": self.instance_id,
            "instance_name": self.instance_name,
            "instance_seed": self.instance_seed,
            "trajectory_source": self.trajectory_source,
            "benders_iteration": self.benders_iteration,
            "num_benders_cuts": self.num_benders_cuts,
        }
        base.update(self.fields)
        return base


@dataclass
class RunRecord:
    """Benders run 하나의 요약 (``record_type = benders_run``)."""

    instance_id: str
    instance_name: str
    instance_seed: int
    trajectory_source: str
    fields: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """평탄화된 결과 dict 를 반환한다."""
        base = {
            "record_type": RECORD_TYPE_RUN,
            "instance_id": self.instance_id,
            "instance_name": self.instance_name,
            "instance_seed": self.instance_seed,
            "trajectory_source": self.trajectory_source,
        }
        base.update(self.fields)
        return base


@dataclass
class BendersTrace:
    """하나의 Benders run 전체 기록."""

    iterations: list[IterationRecord] = field(default_factory=list)
    run: RunRecord | None = None
    cut_snapshots: list[dict[str, Any]] = field(default_factory=list)

    def add_iteration(self, record: IterationRecord) -> None:
        """Iteration 기록을 추가한다."""
        self.iterations.append(record)

    def to_records(self) -> list[dict[str, Any]]:
        """모든 기록을 dict 목록으로 변환한다."""
        out = [it.as_record() for it in self.iterations]
        if self.run is not None:
            out.append(self.run.as_record())
        return out
