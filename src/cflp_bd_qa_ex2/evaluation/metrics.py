"""Solution quality metric.

**분리해서 저장해야 하는 값들**(명세 20절):

    QUBO energy / decoded MP objective / true SP value /
    original CFLP objective / continuous-MP certificate /
    encoded-MP optimum / QUBO ground state

MP 에는 x 가 없으므로 ``decoded_x`` 를 MP metric 으로 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..status import Status


def optimality_gap_percent(candidate: float, ground_truth: float) -> float:
    """``(candidate - ground_truth) / |ground_truth| * 100`` 을 계산한다."""
    return (candidate - ground_truth) / abs(ground_truth) * 100.0


def compute_gap(
    candidate_objective: float | None,
    ground_truth_objective: float | None,
    ground_truth_status: str | Status,
) -> float | None:
    """Ground truth 가 ``OPTIMAL`` 일 때만 gap 을 계산한다.

    Time limit 으로 얻은 incumbent 는 ground truth 로 쓰지 않으므로
    그 경우 ``None`` 을 반환한다.
    """
    if candidate_objective is None or ground_truth_objective is None:
        return None
    if str(ground_truth_status) != str(Status.OPTIMAL):
        return None
    return optimality_gap_percent(candidate_objective, ground_truth_objective)


def discretization_effect(
    continuous_mp_objective: float | None, encoded_mp_objective: float | None
) -> dict[str, Any]:
    """Continuous MP 와 encoded MP 의 차이(순수 discretization 영향)를 계산한다."""
    if continuous_mp_objective is None or encoded_mp_objective is None:
        return {"discretization_absolute": None, "discretization_relative": None}
    diff = encoded_mp_objective - continuous_mp_objective
    scale = max(1e-12, abs(continuous_mp_objective))
    return {
        "discretization_absolute": diff,
        "discretization_relative": diff / scale,
    }


def embedding_success_rate(records: list[dict[str, Any]]) -> float:
    """Embedding trial 목록에서 성공률을 계산한다."""
    if not records:
        return float("nan")
    return sum(1 for r in records if r.get("success")) / len(records)


def summarize_iteration_growth(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Iteration 에 따른 QUBO 성장 요약."""
    if not records:
        return {}
    iterations = np.array([r["benders_iteration"] for r in records], dtype=float)
    variables = np.array([r.get("logical_variables", np.nan) for r in records], dtype=float)
    edges = np.array([r.get("logical_edges", np.nan) for r in records], dtype=float)
    valid = ~np.isnan(variables)
    slope = float("nan")
    if valid.sum() >= 2:
        slope = float(np.polyfit(iterations[valid], variables[valid], 1)[0])
    return {
        "max_iteration": int(iterations.max()),
        "logical_variables_first": float(variables[0]),
        "logical_variables_last": float(variables[-1]),
        "logical_edges_first": float(edges[0]),
        "logical_edges_last": float(edges[-1]),
        "logical_variables_per_iteration": slope,
    }
