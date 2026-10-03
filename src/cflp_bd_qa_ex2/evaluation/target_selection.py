"""실험 대상 인스턴스 선택과 topology 간 일관성 검사.

topology 비교(Pegasus vs Zephyr)가 성립하려면 **두 실행의 대상 인스턴스가
완전히 같아야 한다.** 한쪽만 줄이면 topology 효과와 instance 효과가 섞인다.

선택 로직을 notebook 마다 따로 두면 두 notebook 이 조용히 갈라진다.
그래서 선택과 검사를 이 모듈 한 곳에 둔다(명세 24절).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from ..data.generator import Instance

# 두 topology notebook 의 **공통 기본값**. 한쪽만 바꾸면 비교가 깨진다.
DEFAULT_CONTROLLED_SIZES: list[str] | None = ["25x50"]
DEFAULT_CONTROLLED_SEEDS: list[int] | None = [400]
DEFAULT_E2E_SIZES: list[str] | None = ["25x50"]
DEFAULT_E2E_SEEDS: list[int] | None = [400]


def pick_instances(
    instances: Iterable[Instance],
    sizes: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[str]:
    """크기 label 과 seed 로 instance_id 목록을 고른다.

    두 조건은 AND 로 적용된다. ``sizes=["8x25"]`` 와 ``seeds=[100]`` 처럼
    교집합이 비는 조합을 주면 빈 목록이 나오므로 주의한다.

    Args:
        instances: 후보 인스턴스.
        sizes: 남길 크기 label. ``None`` 이면 제한하지 않는다.
        seeds: 남길 seed. ``None`` 이면 제한하지 않는다.

    Returns:
        ``n_facilities * n_customers`` 오름차순 instance_id 목록.
        작은 것부터 실행해야 중간에 멈춰도 결과가 남는다.
    """
    chosen = [
        inst
        for inst in instances
        if (sizes is None or inst.name in sizes)
        and (seeds is None or inst.seed in seeds)
    ]
    chosen.sort(key=lambda i: (i.n_facilities * i.n_customers, i.instance_id))
    return [inst.instance_id for inst in chosen]


def describe_selection(
    instances: Iterable[Instance],
    sizes: Sequence[str] | None,
    seeds: Sequence[int] | None,
    label: str,
) -> dict[str, Any]:
    """선택 결과를 설명한다. 교집합이 비는 조합을 경고로 알린다."""
    instances = list(instances)
    chosen = pick_instances(instances, sizes, seeds)
    warnings: list[str] = []
    if not chosen:
        warnings.append(
            f"{label}: 선택된 인스턴스가 없다. sizes={sizes}, seeds={seeds} 의 "
            "교집합이 비어 있는지 확인하라"
        )
    elif sizes is not None and seeds is not None:
        missing = [s for s in sizes if not any(i.split("_")[0] == s for i in chosen)]
        if missing:
            warnings.append(
                f"{label}: 크기 {missing} 가 선택되지 않았다. "
                f"seeds={seeds} 와 교집합이 없다"
            )
    return {
        "label": label,
        "instance_ids": chosen,
        "num_selected": len(chosen),
        "sizes": list(sizes) if sizes else None,
        "seeds": list(seeds) if seeds else None,
        "warnings": warnings,
    }


def compare_with_other_topology(
    results_dir: str | Path,
    this_topology: str,
    instance_ids: Sequence[str],
    configuration_hash: str | None = None,
) -> dict[str, Any]:
    """다른 topology 가 이미 돌린 대상과 같은지 확인한다.

    Pegasus 와 Zephyr 의 대상이 다르면 두 결과를 topology 차이로 해석할 수 없다.
    이미 저장된 `qa_runs_<topology>.csv` 를 읽어 비교한다.

    Args:
        results_dir: ``results/qa`` 경로.
        this_topology: 지금 실행하려는 topology.
        instance_ids: 이번에 돌릴 instance_id 목록.
        configuration_hash: 비교를 같은 설정으로 한정한다.

    Returns:
        ``status``, ``message``, ``other_topology``, ``only_here``, ``only_there``.
        비교할 이전 결과가 없으면 ``status`` 가 ``"no_reference"`` 이다.
    """
    import pandas as pd

    results_dir = Path(results_dir)
    others = [
        p
        for p in results_dir.glob("qa_runs_*.csv")
        if not p.stem.endswith(f"_{this_topology}")
    ]
    if not others:
        return {
            "status": "no_reference",
            "message": "비교할 다른 topology 결과가 없다 (첫 실행으로 보인다)",
            "other_topology": None,
            "only_here": [],
            "only_there": [],
        }

    mismatches = []
    for path in others:
        frame = pd.read_csv(path)
        if configuration_hash and "configuration_hash" in frame.columns:
            frame = frame[frame["configuration_hash"] == configuration_hash]
        if frame.empty or "instance_id" not in frame.columns:
            continue
        there = set(frame["instance_id"])
        here = set(instance_ids)
        other = path.stem.replace("qa_runs_", "")
        if here != there:
            mismatches.append(
                {
                    "other_topology": other,
                    "only_here": sorted(here - there),
                    "only_there": sorted(there - here),
                }
            )

    if not mismatches:
        return {
            "status": "match",
            "message": "다른 topology 와 대상 인스턴스가 일치한다",
            "other_topology": None,
            "only_here": [],
            "only_there": [],
        }

    first = mismatches[0]
    return {
        "status": "mismatch",
        "message": (
            f"대상 불일치: '{first['other_topology']}' 와 인스턴스 집합이 다르다. "
            "이 상태로 비교하면 topology 효과와 instance 효과가 섞인다"
        ),
        **first,
    }
