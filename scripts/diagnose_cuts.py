"""Cut signature 진단.

duplicate cut 이 **왜** 생겼는지를 termination_status 와 독립적으로 분석한다.

두 경우를 구분한다.

1. 같은 y 가 재방문되어 같은 cut 이 나온 경우 (``repeated_y``)
2. 서로 다른 y 가 같은 dual 을 주어 같은 cut 이 나온 경우 (``distinct_y_same_cut``)

2번은 SP optimal face 가 같은 서로 다른 y 에서 발생할 수 있다.
**이 스크립트는 관측만 하며 원인을 단정하지 않는다.**
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from cflp_bd_qa_ex2.config.loader import load_config
from cflp_bd_qa_ex2.evaluation.results import (
    MERGE_KEY_MISSING,
    attach_provenance,
    load_qa_results,
    save_records,
)
from cflp_bd_qa_ex2.logging_utils import setup_logging


GROUP_FIELDS: tuple[str, ...] = (
    "instance_id",
    "trajectory_source",
    "graph_id",
    "topology_label",
    "configuration_hash",
)
"""진단 그룹 key.

``instance_id`` 와 ``trajectory_source`` 만으로 묶으면 같은 인스턴스의
Pegasus 와 Zephyr 가 하나의 trajectory 처럼 합쳐진다. 서로 다른 설정의
결과도 섞인다. 따라서 ``graph_id``/``topology_label``/``configuration_hash``
까지 포함한다. 해당 열이 없는 결과(Gurobi, SA)는 빈 문자열로 채운다.
"""

MISSING = MERGE_KEY_MISSING
"""그룹 key 열이 없는 결과(Gurobi/SA)를 채우는 값.

빈 문자열을 쓰면 CSV 로 저장했다가 다시 읽을 때 ``NaN`` 이 되어
merge key 가 달라지고 같은 행이 중복 저장된다.
그래서 왕복해도 문자열로 남는 sentinel 을 쓴다.
"""


def diagnose(frame: pd.DataFrame) -> list[dict]:
    """iteration 기록에서 cut signature 재사용 양상을 진단한다.

    Args:
        frame: Gurobi / SA / QA 의 iteration 기록을 합친 DataFrame.

    Returns:
        그룹별 진단 레코드 목록. 같은 인스턴스라도 topology 나 설정이 다르면
        **서로 다른 행**으로 나뉜다.
    """
    frame = frame.copy()
    for field in GROUP_FIELDS:
        if field not in frame.columns:
            frame[field] = MISSING
        else:
            frame[field] = frame[field].fillna(MISSING)

    rows = []
    for key, group in frame.groupby(list(GROUP_FIELDS), dropna=False):
        group = group.sort_values("benders_iteration")
        signatures = group["cut_signature"].dropna().tolist()

        # trajectory 에 따라 y 가 담기는 열이 다르다.
        # SA/QA 는 decoded_y, gurobi_controlled 는 continuous_mp_y 에 들어간다.
        y_values = []
        for _, row in group.iterrows():
            value = row.get("decoded_y")
            if value is None or (isinstance(value, float) and pd.isna(value)):
                value = row.get("continuous_mp_y")
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                y_values.append(str(value))

        distinct_cuts = len(set(signatures))
        distinct_y = len(set(y_values)) if y_values else None

        pattern = "none"
        if distinct_cuts < len(signatures):
            if distinct_y is not None and distinct_y > distinct_cuts:
                pattern = "distinct_y_same_cut"
            else:
                pattern = "repeated_y"

        rows.append(
            {
                "record_type": "cut_signature_diagnosis",
                **dict(zip(GROUP_FIELDS, key)),
                "num_iterations": len(group),
                "num_cuts_generated": len(signatures),
                "num_distinct_cuts": distinct_cuts,
                "num_distinct_y": distinct_y,
                "num_duplicate_signatures": len(signatures) - distinct_cuts,
                "duplicate_pattern": pattern,
            }
        )
    return rows


def main() -> None:
    """SA / Gurobi / QA iteration 기록을 읽어 진단 결과를 저장한다."""
    parser = argparse.ArgumentParser(description="cut signature 진단")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    sources = [
        Path("results/benders/benders_iterations.csv"),
        Path("results/sa/sa_iterations.csv"),
    ]
    frames = [pd.read_csv(p) for p in sources if p.exists()]

    # QA 결과는 topology 접미사가 붙는다 (qa_iterations_pegasus.csv 등).
    # 구형 무접미사 파일만 읽으면 QA 가 진단에서 통째로 빠진다.
    qa_frame = load_qa_results(Path("results/qa"), "qa_iterations")
    if not qa_frame.empty:
        frames.append(qa_frame)

    frames = [f for f in frames if not f.empty]
    if not frames:
        logger.warning("진단할 iteration 기록이 없다")
        return

    rows = diagnose(pd.concat(frames, ignore_index=True))
    save_records(
        attach_provenance(rows, cfg),
        "results/benders",
        "cut_signature_diagnosis",
        cfg["output"]["formats"],
        merge_keys=GROUP_FIELDS,
    )
    frame = pd.DataFrame(rows)
    logger.info("진단 결과 %d행", len(frame))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
