"""Minorminer 기반 embedding 탐색과 metric 수집.

파일럿에서는 매 iteration 의 QUBO 를 각 seed 로 **처음부터 독립적으로**
임베딩한다. 다음은 사용하지 않는다.

    fixed_chains / initial_chains / suspend_chains
    incremental embedding / previous-iteration embedding reuse

``minorminer`` 의 실패는 embeddability 의 수학적 부재를 증명하지 않는다.
결과 설명에서는 반드시 "주어진 search budget 에서 embedding 을 찾지 못함"
이라고 표현한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from ..status import NA, Status
from .targets import TargetGraph
from .validation import full_validation


@dataclass
class EmbeddingTrial:
    """Embedding 시도 하나의 결과."""

    success: bool
    embedding_status: Status
    embedding_seed: int
    embedding_time: float
    embedding: dict[Any, list[int]] | None = None
    message: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict."""
        base = {
            "success": self.success,
            "embedding_status": str(self.embedding_status),
            "embedding_seed": self.embedding_seed,
            "embedding_time": self.embedding_time,
            "embedding_message": self.message,
        }
        base.update(self.fields)
        return base


def chain_metrics(embedding: dict[Any, list[int]], logical_variables: int) -> dict[str, Any]:
    """Chain 과 physical qubit metric 을 계산한다."""
    lengths = np.array([len(chain) for chain in embedding.values()], dtype=float)
    physical = int(lengths.sum())
    return {
        "physical_qubits": physical,
        "min_chain_length": int(lengths.min()),
        "mean_chain_length": float(lengths.mean()),
        "median_chain_length": float(np.median(lengths)),
        "max_chain_length": int(lengths.max()),
        "std_chain_length": float(lengths.std(ddof=0)),
        "embedding_overhead": physical / logical_variables if logical_variables else float("nan"),
    }


def failed_chain_metrics() -> dict[str, Any]:
    """실패한 trial 의 chain metric 을 ``NA`` 로 채운다."""
    return {
        "physical_qubits": NA,
        "min_chain_length": NA,
        "mean_chain_length": NA,
        "median_chain_length": NA,
        "max_chain_length": NA,
        "std_chain_length": NA,
        "embedding_overhead": NA,
        "physical_qubits_per_logical_edge": NA,
    }


def find_embedding_with_seed(
    source: nx.Graph,
    target: TargetGraph,
    seed: int,
    timeout: int,
    tries: int,
) -> EmbeddingTrial:
    """단일 seed 로 embedding 을 탐색하고 검증한다.

    Args:
        source: 모든 logical variable 을 포함한 source graph.
        target: target graph.
        seed: ``minorminer`` random seed.
        timeout: 초 단위 제한.
        tries: 시도 횟수.

    Returns:
        :class:`EmbeddingTrial`.
    """
    import minorminer

    start = time.perf_counter()
    try:
        embedding = minorminer.find_embedding(
            source,
            target.graph,
            random_seed=seed,
            timeout=timeout,
            tries=tries,
        )
    except Exception as exc:  # pragma: no cover - 환경 의존
        return EmbeddingTrial(
            success=False,
            embedding_status=Status.EMBEDDING_FAILED,
            embedding_seed=seed,
            embedding_time=time.perf_counter() - start,
            message=f"minorminer 예외: {exc}",
            fields=failed_chain_metrics(),
        )
    elapsed = time.perf_counter() - start

    if not embedding:
        return EmbeddingTrial(
            success=False,
            embedding_status=Status.EMBEDDING_FAILED,
            embedding_seed=seed,
            embedding_time=elapsed,
            message="주어진 search budget 에서 embedding 을 찾지 못함",
            fields=failed_chain_metrics(),
        )

    embedding = {variable: list(chain) for variable, chain in embedding.items()}
    valid, reason = full_validation(embedding, source, target.graph)
    if not valid:
        return EmbeddingTrial(
            success=False,
            embedding_status=Status.EMBEDDING_INVALID,
            embedding_seed=seed,
            embedding_time=elapsed,
            message=reason,
            fields=failed_chain_metrics(),
        )

    metrics = chain_metrics(embedding, source.number_of_nodes())
    edges = source.number_of_edges()
    metrics["physical_qubits_per_logical_edge"] = (
        metrics["physical_qubits"] / edges if edges else NA
    )

    return EmbeddingTrial(
        success=True,
        embedding_status=Status.SUCCESS,
        embedding_seed=seed,
        embedding_time=elapsed,
        embedding=embedding,
        fields=metrics,
    )


def run_embedding_trials(
    source: nx.Graph,
    target: TargetGraph,
    cfg: dict,
) -> list[EmbeddingTrial]:
    """설정된 모든 seed 로 embedding 을 시도한다.

    실패한 trial 도 반드시 결과에 남긴다.
    """
    ecfg = cfg["embedding"]
    return [
        find_embedding_with_seed(
            source, target, int(seed), int(ecfg["timeout"]), int(ecfg["tries"])
        )
        for seed in ecfg["seeds"]
    ]
