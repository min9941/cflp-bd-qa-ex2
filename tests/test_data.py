"""데이터 생성 재현성과 schema 테스트."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflp_bd_qa_ex2.data.generator import DRAW_ORDER, generate_instance
from cflp_bd_qa_ex2.data.io import (
    instance_from_dict,
    instance_to_dict,
    validate_instance_payload,
)


def test_generation_is_deterministic(pilot_config):
    """같은 seed 는 항상 같은 인스턴스를 만든다."""
    a = generate_instance("4x12", 4, 12, 100, pilot_config)
    b = generate_instance("4x12", 4, 12, 100, pilot_config)
    assert np.array_equal(a.demand, b.demand)
    assert np.array_equal(a.capacity, b.capacity)
    assert np.allclose(a.fixed_cost, b.fixed_cost)
    assert np.allclose(a.transport_cost, b.transport_cost)


def test_different_seeds_differ(pilot_config):
    """다른 seed 는 다른 인스턴스를 만든다."""
    a = generate_instance("4x12", 4, 12, 100, pilot_config)
    b = generate_instance("4x12", 4, 12, 101, pilot_config)
    assert not np.array_equal(a.demand, b.demand)


def test_total_capacity_equality(pilot_config):
    """총용량은 ceil(1.5 * D) 와 정확히 일치한다."""
    for name, nf, nc, seed in [("4x12", 4, 12, 100), ("8x25", 8, 25, 200), ("16x50", 16, 50, 300)]:
        inst = generate_instance(name, nf, nc, seed, pilot_config)
        expected = math.ceil(1.5 * inst.total_demand)
        assert inst.total_capacity == expected


def test_demand_is_positive_integer(pilot_config):
    """demand 는 1 이상의 정수이다."""
    inst = generate_instance("8x25", 8, 25, 200, pilot_config)
    assert inst.demand.dtype.kind == "i"
    assert inst.demand.min() >= 1


def test_demand_uses_standard_deviation_not_variance(pilot_config):
    """demand 의 5는 표준편차로 해석된다 (분산 아님)."""
    inst = generate_instance("large", 2, 4000, 999, pilot_config)
    assert 4.3 < float(inst.demand.std(ddof=1)) < 5.7


def test_capacity_remainder_tie_breaks_by_index():
    """remainder 가 같으면 index 가 작은 시설부터 배분한다."""
    from cflp_bd_qa_ex2.data.generator import _integerize_capacity

    raw = np.array([1.0, 1.0, 1.0])
    result = _integerize_capacity(raw, 10)
    assert result.sum() == 10
    assert result[0] >= result[1] >= result[2]


def test_draw_order_documented():
    """draw 순서가 문서화되어 있다."""
    assert DRAW_ORDER == (
        "facility_locations",
        "customer_locations",
        "demand_raw",
        "capacity_raw",
        "fixed_cost_base",
        "fixed_cost_slope",
    )


def test_schema_roundtrip(small_instance):
    """저장-복원 후에도 값이 보존된다."""
    payload = instance_to_dict(small_instance)
    validate_instance_payload(payload)
    restored = instance_from_dict(payload)
    assert np.array_equal(restored.demand, small_instance.demand)
    assert np.allclose(restored.transport_cost, small_instance.transport_cost)


def test_schema_rejects_bad_payload(small_instance):
    """schema 위반을 검출한다."""
    payload = instance_to_dict(small_instance)
    payload["demand"] = [0] * small_instance.n_customers
    with pytest.raises(ValueError):
        validate_instance_payload(payload)
