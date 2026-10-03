"""Embedding 유효성 검증.

**성공 여부를 non-empty mapping 으로 판단하지 않는다.**
다음을 모두 확인한다(명세 16절).

- 모든 logical variable 에 non-empty chain 존재
- chain 간 physical qubit 중복 없음
- 모든 physical qubit 가 target graph 에 존재
- 각 chain 내부가 연결됨
- 모든 logical edge 를 실현하는 physical coupler 존재
"""

from __future__ import annotations

from typing import Any

import networkx as nx


class EmbeddingInvalid(ValueError):
    """Embedding 이 유효하지 않을 때 발생하는 예외 (``EMBEDDING_INVALID``)."""


def validate_embedding(
    embedding: dict[Any, list[int]],
    source: nx.Graph,
    target: nx.Graph,
) -> tuple[bool, str]:
    """Embedding 을 검증한다.

    Args:
        embedding: logical variable -> physical qubit chain.
        source: 모든 logical variable 을 포함한 source graph
            (isolated variable 포함).
        target: target working graph.

    Returns:
        ``(valid, reason)``. 유효하면 ``reason`` 은 빈 문자열이다.
    """
    if not embedding:
        return False, "embedding 이 비어 있다"

    missing = [v for v in source.nodes() if v not in embedding or len(embedding[v]) == 0]
    if missing:
        return False, f"chain 이 없는 logical variable {len(missing)}개 (예: {missing[:5]})"

    extra = [v for v in embedding if v not in source.nodes()]
    if extra:
        return False, f"source 에 없는 변수가 embedding 에 있다: {extra[:5]}"

    used: set[int] = set()
    for variable, chain in embedding.items():
        chain_set = set(chain)
        if len(chain_set) != len(chain):
            return False, f"chain 내부에 중복 qubit 이 있다: {variable}"
        overlap = used & chain_set
        if overlap:
            return False, f"chain 간 physical qubit 중복: {variable} 에서 {sorted(overlap)[:5]}"
        used |= chain_set

        for qubit in chain:
            if qubit not in target:
                return False, f"target graph 에 없는 physical qubit: {qubit} ({variable})"

        if not nx.is_connected(target.subgraph(chain)):
            return False, f"chain 이 연결되어 있지 않다: {variable}"

    for a, b in source.edges():
        if a == b:
            continue
        has_coupler = any(
            target.has_edge(qa, qb) for qa in embedding[a] for qb in embedding[b]
        )
        if not has_coupler:
            return False, f"logical edge ({a}, {b}) 를 실현하는 physical coupler 가 없다"

    return True, ""


def verify_with_ocean(
    embedding: dict[Any, list[int]], source: nx.Graph, target: nx.Graph
) -> tuple[bool, str]:
    """D-Wave Ocean 의 ``verify_embedding`` 으로 교차 검증한다.

    Ocean 은 isolated source node 를 확인하지 않을 수 있으므로
    :func:`validate_embedding` 과 함께 사용한다.
    """
    try:
        from dwave.embedding import verify_embedding as _verify

        _verify(embedding, source, target)
        return True, ""
    except Exception as exc:
        return False, f"ocean verify_embedding 실패: {exc}"


def full_validation(
    embedding: dict[Any, list[int]], source: nx.Graph, target: nx.Graph
) -> tuple[bool, str]:
    """자체 검증과 Ocean 검증을 모두 통과해야 유효로 판정한다."""
    ok, reason = validate_embedding(embedding, source, target)
    if not ok:
        return False, reason
    ok_ocean, reason_ocean = verify_with_ocean(embedding, source, target)
    if not ok_ocean:
        return False, reason_ocean
    return True, ""
