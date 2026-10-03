"""Encoding 테스트: theta endpoint, capacity slack 정확성, cut slack bit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflp_bd_qa_ex2.encoding.pure_exponential import (
    EncodingError,
    PureExponentialEncoding,
    bits_for_range,
)
from cflp_bd_qa_ex2.encoding.slack import (
    CutSlackRangeInvalid,
    build_capacity_slack_encoding,
    build_cut_slack_encoding,
    cut_slack_max,
)
from cflp_bd_qa_ex2.encoding.theta import build_theta_encoding


def test_theta_endpoints(small_instance):
    """모든 bit 가 0이면 theta_L, 모두 1이면 theta_U 이다."""
    enc = build_theta_encoding(small_instance.objective_upper_bound, 10)
    assert enc.decode([0] * 10) == pytest.approx(enc.theta_lower_bound)
    assert enc.decode([1] * 10) == pytest.approx(enc.theta_upper_bound)


def test_theta_delta_definition(small_instance):
    """delta = (U - L) / (2^b - 1)."""
    bits = 10
    enc = build_theta_encoding(small_instance.objective_upper_bound, bits)
    expected = (enc.theta_upper_bound - enc.theta_lower_bound) / (2**bits - 1)
    assert enc.theta_delta == pytest.approx(expected)


def test_theta_is_not_truncated(small_instance):
    """마지막 bit 를 줄이는 truncated encoding 을 쓰지 않는다."""
    bits = 8
    enc = build_theta_encoding(small_instance.objective_upper_bound, bits)
    coefficients = enc.encoding.coefficients()
    assert len(coefficients) == bits
    assert coefficients[-1] == pytest.approx(enc.theta_delta * 2 ** (bits - 1))


def test_theta_range_invalid():
    """상한이 0 이하면 THETA_RANGE_INVALID 에 해당하는 예외."""
    with pytest.raises(EncodingError):
        build_theta_encoding(0.0, 10)
    with pytest.raises(EncodingError):
        build_theta_encoding(100.0, 0)


def test_capacity_slack_is_exact(small_instance):
    """capacity slack 은 [0, S_cap_max] 의 모든 정수를 정확히 표현한다."""
    enc = build_capacity_slack_encoding(
        small_instance.total_capacity, small_instance.total_demand
    )
    assert enc.capacity_slack_max == small_instance.capacity_slack_max
    assert enc.encoding.max_value >= enc.capacity_slack_max
    for value in range(0, enc.capacity_slack_max + 1):
        bits = enc.encoding.encode_nearest(float(value))
        assert enc.decode(bits) == pytest.approx(float(value))


def test_capacity_slack_bit_count(small_instance):
    """b_cap = ceil(log2(S_cap_max + 1))."""
    enc = build_capacity_slack_encoding(
        small_instance.total_capacity, small_instance.total_demand
    )
    expected = math.ceil(math.log2(small_instance.capacity_slack_max + 1))
    assert enc.capacity_slack_bits == expected


def test_capacity_slack_rejects_nonunit_delta(small_instance):
    """delta != 1 이면 거부한다 (근사 금지)."""
    with pytest.raises(EncodingError):
        build_capacity_slack_encoding(
            small_instance.total_capacity, small_instance.total_demand, delta=2.0
        )


def test_cut_slack_bit_formula(small_instance):
    """b_k = ceil(log2(S_k_max / delta_cut + 1))."""
    theta = build_theta_encoding(small_instance.objective_upper_bound, 10)
    capacity = small_instance.capacity.astype(float)
    v = np.full(small_instance.n_facilities, 3.0)
    u_sum = 2000.0
    enc = build_cut_slack_encoding(0, theta.theta_upper_bound, capacity, v, u_sum,
                                   theta.theta_delta, 0.25)
    s_max = cut_slack_max(theta.theta_upper_bound, capacity, v, u_sum)
    expected = math.ceil(math.log2(s_max / (0.25 * theta.theta_delta) + 1))
    assert enc.cut_slack_bits == expected
    assert enc.cut_slack_delta == pytest.approx(0.25 * theta.theta_delta)


def test_cut_slack_zero_range_has_no_bits(small_instance):
    """S_k_max = 0 이면 slack bit 를 만들지 않는다."""
    theta = build_theta_encoding(small_instance.objective_upper_bound, 10)
    capacity = small_instance.capacity.astype(float)
    v = np.zeros(small_instance.n_facilities)
    enc = build_cut_slack_encoding(0, theta.theta_upper_bound, capacity, v,
                                   theta.theta_upper_bound, theta.theta_delta, 0.25)
    assert enc.cut_slack_bits == 0


def test_cut_slack_small_negative_is_clamped(small_instance):
    """-1e-6 <= S_k_max < 0 은 0으로 처리하고 표시를 남긴다."""
    theta = build_theta_encoding(small_instance.objective_upper_bound, 10)
    capacity = small_instance.capacity.astype(float)
    v = np.zeros(small_instance.n_facilities)
    enc = build_cut_slack_encoding(
        0, theta.theta_upper_bound, capacity, v,
        theta.theta_upper_bound + 5e-7, theta.theta_delta, 0.25,
    )
    assert enc.cut_slack_bits == 0
    assert enc.clamped_from_negative is True


def test_cut_slack_large_negative_raises(small_instance):
    """S_k_max < -1e-6 이면 CUT_SLACK_RANGE_INVALID."""
    theta = build_theta_encoding(small_instance.objective_upper_bound, 10)
    capacity = small_instance.capacity.astype(float)
    v = np.zeros(small_instance.n_facilities)
    with pytest.raises(CutSlackRangeInvalid):
        build_cut_slack_encoding(
            0, theta.theta_upper_bound, capacity, v,
            theta.theta_upper_bound + 1.0, theta.theta_delta, 0.25,
        )


def test_cut_slack_bits_not_capped(small_instance):
    """bit 수를 임의로 cap 하지 않는다 (매우 작은 delta 에서도)."""
    theta = build_theta_encoding(small_instance.objective_upper_bound, 20)
    capacity = small_instance.capacity.astype(float)
    v = np.zeros(small_instance.n_facilities)
    enc = build_cut_slack_encoding(0, theta.theta_upper_bound, capacity, v, 0.0,
                                   theta.theta_delta, 0.01)
    assert enc.cut_slack_bits > 20


def test_bits_for_range_edges():
    """bits_for_range 의 경계 동작."""
    assert bits_for_range(0.0, 1.0) == 0
    assert bits_for_range(1.0, 1.0) == 1
    assert bits_for_range(3.0, 1.0) == 2
    assert bits_for_range(4.0, 1.0) == 3
    with pytest.raises(EncodingError):
        bits_for_range(-1.0, 1.0)


def test_pure_exponential_roundtrip():
    """encode/decode 왕복이 격자점에서 정확하다."""
    enc = PureExponentialEncoding(n_bits=6, delta=0.5, offset=2.0, label="z")
    for steps in range(2**6):
        value = 2.0 + 0.5 * steps
        assert enc.decode(enc.encode_nearest(value)) == pytest.approx(value)
