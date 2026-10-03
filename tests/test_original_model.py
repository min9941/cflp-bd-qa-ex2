"""Original CFLP 모델 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from cflp_bd_qa_ex2.models.cflp import evaluate_original_objective, solve_original_cflp
from cflp_bd_qa_ex2.status import Status

pytestmark = pytest.mark.filterwarnings("ignore")


def test_original_cflp_optimal(small_instance, pilot_config):
    """4x12 는 OPTIMAL 로 풀린다."""
    outcome = solve_original_cflp(small_instance, pilot_config)
    assert outcome.status is Status.OPTIMAL
    assert outcome.objective is not None and outcome.objective > 0


def test_open_capacity_covers_demand(small_instance, pilot_config):
    """최적해에서 열린 시설의 총용량은 총수요 이상이다."""
    outcome = solve_original_cflp(small_instance, pilot_config)
    y = np.asarray(outcome.payload["y"], dtype=int)
    assert float(small_instance.capacity @ y) >= small_instance.total_demand - 1e-9


def test_closed_facility_receives_no_flow(small_instance, pilot_config):
    """y_j = 0 이면 x_ij = 0 이다 (x_ij <= y_j 없이도 성립)."""
    outcome = solve_original_cflp(small_instance, pilot_config)
    y = np.asarray(outcome.payload["y"], dtype=int)
    x = np.asarray(outcome.payload["x"], dtype=float)
    for j in range(small_instance.n_facilities):
        if y[j] == 0:
            assert x[:, j].max() <= 1e-9


def test_evaluate_original_objective(small_instance):
    """목적값 계산이 f@y + Q(y) 와 같다."""
    y = np.ones(small_instance.n_facilities, dtype=int)
    value = evaluate_original_objective(small_instance, y, 100.0)
    assert value == pytest.approx(float(small_instance.fixed_cost.sum()) + 100.0)
