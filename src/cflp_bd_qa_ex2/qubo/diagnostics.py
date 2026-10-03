"""QUBO 진단 metric.

**Coefficient range 는 차이가 아니라 비율이다.**

    R_Q = max_{Q_ab != 0} |Q_ab| / min_{Q_ab != 0} |Q_ab|

logical QUBO, Ising h/J, embedded physical problem 의 coefficient 범위를
서로 섞지 않는다. 이 모듈은 **logical QUBO** 만 다룬다.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from .builder import MasterQUBO, ROLE_CAPACITY_SLACK, ROLE_CUT_SLACK, ROLE_THETA, ROLE_Y


def logical_graph(qubo: MasterQUBO) -> nx.Graph:
    """QUBO 의 source graph 를 만든다.

    **quadratic edge 가 없는 isolated logical variable 도 반드시 포함한다.**
    이를 빠뜨리면 embedding 이 해당 변수를 누락한 채 성공한 것처럼 보인다.
    """
    graph = nx.Graph()
    graph.add_nodes_from(qubo.variable_labels)
    for (a, b) in qubo.quadratic:
        graph.add_edge(a, b)
    return graph


def qubo_metrics(qubo: MasterQUBO) -> dict[str, Any]:
    """logical QUBO 의 크기/밀도/계수 범위 metric 을 계산한다.

    Returns:
        명세 12절과 17절에서 요구하는 metric dict.
    """
    n = qubo.num_variables
    n_quadratic = len(qubo.quadratic)
    max_edges = n * (n - 1) / 2
    density = (n_quadratic / max_edges) if max_edges > 0 else 0.0

    magnitudes = [abs(v) for v in qubo.linear.values() if v != 0.0]
    magnitudes += [abs(v) for v in qubo.quadratic.values() if v != 0.0]

    if magnitudes:
        min_abs = min(magnitudes)
        max_abs = max(magnitudes)
        ratio = max_abs / min_abs
        log10_ratio = math.log10(ratio) if ratio > 0 else 0.0
    else:  # pragma: no cover - 빈 QUBO 는 발생하지 않음
        min_abs = max_abs = ratio = log10_ratio = 0.0

    return {
        "qubo_num_variables": n,
        "qubo_num_quadratic_terms": n_quadratic,
        "qubo_density": density,
        "qubo_min_abs_coefficient": min_abs,
        "qubo_max_abs_coefficient": max_abs,
        "qubo_coefficient_range": ratio,
        "qubo_log10_coefficient_range": log10_ratio,
        "qubo_offset": qubo.offset,
        "logical_variables": n,
        "logical_edges": n_quadratic,
        "logical_density": density,
    }


def variable_role_table(qubo: MasterQUBO) -> list[dict[str, Any]]:
    """변수 role table 을 만든다 (notebook 03 용)."""
    rows = []
    for name in qubo.variable_labels:
        rows.append(
            {
                "variable": name,
                "role": qubo.variable_roles[name],
                "linear_coefficient": qubo.linear.get(name, 0.0),
                "degree": sum(1 for (a, b) in qubo.quadratic if name in (a, b)),
            }
        )
    return rows


def role_counts(qubo: MasterQUBO) -> dict[str, int]:
    """role 별 변수 개수를 센다."""
    return {
        "num_y_variables": len(qubo.variables_by_role(ROLE_Y)),
        "num_theta_bits": len(qubo.variables_by_role(ROLE_THETA)),
        "num_capacity_slack_bits": len(qubo.variables_by_role(ROLE_CAPACITY_SLACK)),
        "num_cut_slack_bits": len(qubo.variables_by_role(ROLE_CUT_SLACK)),
    }


def source_graph_fingerprint(graph: nx.Graph) -> str:
    """Source graph 의 결정적 지문.

    노드 수, 엣지 수, 정렬된 엣지 목록의 해시를 결합한다.
    """
    import hashlib

    # node label 을 반드시 포함한다. 그렇지 않으면 isolated node 의 이름만 다른
    # 서로 다른 source graph 가 같은 지문을 갖는다.
    nodes = sorted(str(n) for n in graph.nodes())
    edges = sorted(tuple(sorted((str(a), str(b)))) for a, b in graph.edges())
    blob = repr((nodes, edges)).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return f"n{graph.number_of_nodes()}e{graph.number_of_edges()}_{digest}"
