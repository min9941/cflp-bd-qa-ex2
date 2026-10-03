"""성공한 embedding mapping 저장.

파일명에는 최소한 다음을 포함한다(명세 16절).

    instance_id, benders_iteration, topology_label, solver_id 또는 ideal,
    graph_id, embedding_seed, timeout, tries, source_graph_fingerprint

**기존 파일을 덮어쓰지 않는다.** 같은 이름이 있으면 접미사를 붙인다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def embedding_filename(
    instance_id: str,
    benders_iteration: int,
    topology_label: str,
    solver_id: str | None,
    graph_id: str,
    embedding_seed: int,
    timeout: int,
    tries: int,
    source_graph_fingerprint: str,
) -> str:
    """Embedding 파일 이름을 만든다."""
    solver_part = solver_id if solver_id else "ideal"
    return (
        f"{instance_id}__it{benders_iteration:03d}__{topology_label}"
        f"__{solver_part}__{graph_id}__seed{embedding_seed}"
        f"__to{timeout}__tr{tries}__{source_graph_fingerprint}.json"
    )


def save_embedding(
    embedding: dict[Any, list[int]],
    out_dir: str | Path,
    metadata: dict[str, Any],
    filename: str,
) -> Path:
    """Embedding mapping 을 JSON 으로 저장한다 (덮어쓰기 금지).

    Returns:
        실제로 저장된 경로.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    suffix = 1
    while path.exists():
        path = out_dir / f"{Path(filename).stem}__dup{suffix}.json"
        suffix += 1

    payload = {
        "metadata": metadata,
        "embedding": {str(k): list(map(int, v)) for k, v in embedding.items()},
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)
    return path


def load_embedding(path: str | Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """저장된 embedding 을 읽는다."""
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload["embedding"], payload["metadata"]
