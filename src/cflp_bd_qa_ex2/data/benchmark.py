"""Beasley OR-Library CFLP 벤치마크 파일 변환.

대상 포맷은 `cap*.txt` (cap41~cap74, capa/capb/capc 등)이다.

파일 구조::

    m n                     시설 수, 고객 수
    s_1  f_1                시설 1 의 capacity, fixed cost
    ...
    s_m  f_m
    d_1                     고객 1 의 demand
    c_11 c_12 ... c_1m      고객 1 의 **총** 수송비 (수요 전량을 각 시설에서 공급할 때)
    d_2
    c_21 ...
    ...

공백과 줄바꿈은 의미가 없다. 값은 공백으로 구분된 토큰 순서로만 읽는다.

**비용 해석이 핵심이다.** OR-Library 의 `c_ij` 는 고객 `j` 의 수요를 **전량**
시설 `i` 에서 공급하는 총비용이다. 본 프로젝트의 목적함수는
``sum_i sum_j d_i c_ij x_ij`` 이므로 단위비용이 필요하다. 따라서

    unit_cost_ij = total_cost_ij / d_i

로 변환한다. 이렇게 하면 ``d_i * unit_cost_ij == total_cost_ij`` 가 되어
원본 목적함수 값이 보존된다.

좌표는 **존재하지 않는다**. OR-Library 는 비용 행렬을 직접 제공하며
좌표에서 거리를 계산하지 않는다. 일반적인 비용 행렬은 유클리드 거리로
역산되지도 않으므로(삼각부등식 미보장), 좌표는 ``NaN`` 으로 두고
metadata 의 ``has_locations`` 를 ``False`` 로 표시한다.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .generator import Instance


class BenchmarkParseError(ValueError):
    """벤치마크 파일을 해석할 수 없을 때 발생하는 예외."""


def _benchmark_number(path: Path) -> str:
    """파일명에서 벤치마크 식별자를 뽑는다.

    ``cap41.txt`` -> ``"41"``, ``capa.txt`` -> ``"a"``, ``cap131.txt`` -> ``"131"``.

    Raises:
        BenchmarkParseError: ``cap`` 으로 시작하지 않는 파일명인 경우.
    """
    match = re.fullmatch(r"cap[_\-]?([0-9a-zA-Z]+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise BenchmarkParseError(
            f"파일명에서 벤치마크 번호를 찾을 수 없다: {path.name}. "
            "'cap41.txt' 같은 형식이어야 한다"
        )
    return match.group(1).lower()


def parse_or_library_cap(path: str | Path) -> dict[str, Any]:
    """OR-Library cap 파일을 raw 값 dict 로 읽는다.

    Args:
        path: `cap*.txt` 경로.

    Returns:
        ``n_facilities``, ``n_customers``, ``capacity``, ``fixed_cost``,
        ``demand``, ``total_cost`` (shape ``(n_customers, n_facilities)``) 를 담은 dict.

    Raises:
        BenchmarkParseError: 토큰 수가 맞지 않거나 값이 유효하지 않은 경우.
    """
    path = Path(path)
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    if len(tokens) < 2:
        raise BenchmarkParseError(f"{path.name}: 토큰이 너무 적다")

    try:
        n_facilities, n_customers = int(float(tokens[0])), int(float(tokens[1]))
    except ValueError as exc:
        raise BenchmarkParseError(f"{path.name}: 첫 두 토큰이 정수가 아니다") from exc
    if n_facilities < 1 or n_customers < 1:
        raise BenchmarkParseError(f"{path.name}: 크기가 1 미만이다")

    expected = 2 + 2 * n_facilities + n_customers * (1 + n_facilities)
    if len(tokens) != expected:
        raise BenchmarkParseError(
            f"{path.name}: 토큰 수 불일치. 기대 {expected}, 실제 {len(tokens)}. "
            f"({n_facilities}x{n_customers} 포맷이 맞는지 확인하라)"
        )

    cursor = 2
    capacity = np.empty(n_facilities, dtype=float)
    fixed_cost = np.empty(n_facilities, dtype=float)
    for j in range(n_facilities):
        capacity[j] = float(tokens[cursor])
        fixed_cost[j] = float(tokens[cursor + 1])
        cursor += 2

    demand = np.empty(n_customers, dtype=float)
    total_cost = np.empty((n_customers, n_facilities), dtype=float)
    for i in range(n_customers):
        demand[i] = float(tokens[cursor])
        cursor += 1
        total_cost[i, :] = [float(tokens[cursor + k]) for k in range(n_facilities)]
        cursor += n_facilities

    return {
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        "capacity": capacity,
        "fixed_cost": fixed_cost,
        "demand": demand,
        "total_cost": total_cost,
    }


def benchmark_to_instance(path: str | Path) -> Instance:
    """OR-Library cap 파일을 본 프로젝트의 :class:`Instance` 로 변환한다.

    변환 규칙:

    - ``unit_cost_ij = total_cost_ij / d_i`` (목적함수 값이 보존된다)
    - demand, capacity 는 정수로 유지한다. capacity slack 을 ``delta = 1`` 로
      **정확히** 표현하려면 정수여야 한다.
    - 좌표는 ``NaN``. 원본에 좌표가 없다.
    - ``instance_id`` 는 ``{m}x{n}_b{번호}``, ``name`` 은 ``cap{번호}``.
      생성 데이터의 ``16x50`` 과 ``instance_name`` 이 겹치지 않게 한다.

    Args:
        path: `cap*.txt` 경로.

    Returns:
        변환된 :class:`Instance`.

    Raises:
        BenchmarkParseError: demand/capacity 가 정수가 아니거나 음수,
            또는 총용량이 총수요보다 작아 MP 가 infeasible 한 경우.
    """
    path = Path(path)
    raw = parse_or_library_cap(path)
    number = _benchmark_number(path)

    demand_float = raw["demand"]
    capacity_float = raw["capacity"]

    for label, values in (("demand", demand_float), ("capacity", capacity_float)):
        if not np.all(np.isfinite(values)):
            raise BenchmarkParseError(f"{path.name}: {label} 에 유한하지 않은 값이 있다")
        if not np.allclose(values, np.rint(values)):
            raise BenchmarkParseError(
                f"{path.name}: {label} 가 정수가 아니다. "
                "capacity slack 을 delta=1 로 정확히 표현할 수 없다"
            )

    demand = np.rint(demand_float).astype(np.int64)
    capacity = np.rint(capacity_float).astype(np.int64)

    if demand.min() < 1:
        raise BenchmarkParseError(f"{path.name}: demand 는 1 이상이어야 한다")
    if capacity.min() < 0:
        raise BenchmarkParseError(f"{path.name}: capacity 는 음수일 수 없다")

    total_demand = int(demand.sum())
    total_capacity = int(capacity.sum())
    if total_capacity < total_demand:
        raise BenchmarkParseError(
            f"{path.name}: 총용량 {total_capacity} < 총수요 {total_demand}. "
            "aggregate capacity constraint 를 만족할 수 없어 MP 가 infeasible 하다"
        )

    # OR-Library 의 c_ij 는 수요 전량 공급 비용이므로 단위비용으로 나눈다.
    unit_cost = raw["total_cost"] / demand[:, None].astype(float)
    if not np.all(np.isfinite(unit_cost)):
        raise BenchmarkParseError(f"{path.name}: 단위비용에 유한하지 않은 값이 있다")
    if unit_cost.min() < 0:
        raise BenchmarkParseError(f"{path.name}: 단위비용이 음수이다")

    # 왕복 검증: d_i * unit_cost_ij 가 원본 총비용과 일치해야 한다.
    reconstructed = unit_cost * demand[:, None].astype(float)
    max_error = float(np.max(np.abs(reconstructed - raw["total_cost"])))
    scale = max(1.0, float(np.max(np.abs(raw["total_cost"]))))
    if max_error / scale > 1e-12:
        raise BenchmarkParseError(
            f"{path.name}: 단위비용 왕복 오차가 크다 ({max_error:.3e})"
        )

    nan_locations_facility = np.full((raw["n_facilities"], 2), np.nan)
    nan_locations_customer = np.full((raw["n_customers"], 2), np.nan)

    return Instance(
        instance_id=f"{raw['n_facilities']}x{raw['n_customers']}_b{number}",
        name=f"cap{number}",
        seed=-1,
        n_facilities=raw["n_facilities"],
        n_customers=raw["n_customers"],
        facility_locations=nan_locations_facility,
        customer_locations=nan_locations_customer,
        demand=demand,
        capacity=capacity,
        fixed_cost=raw["fixed_cost"].astype(float),
        transport_cost=unit_cost,
        metadata={
            "source": "or_library",
            "source_file": path.name,
            "benchmark_id": number,
            "has_locations": False,
            "locations_note": (
                "OR-Library 는 비용 행렬을 직접 제공하며 좌표가 없다. "
                "일반 비용 행렬은 유클리드 거리로 역산되지 않으므로 NaN 으로 둔다"
            ),
            "cost_note": (
                "원본 c_ij 는 수요 전량 공급 총비용이다. "
                "unit_cost_ij = c_ij / d_i 로 변환해 d_i * unit_cost_ij == c_ij 가 성립한다"
            ),
            "total_demand": total_demand,
            "total_capacity": total_capacity,
            "capacity_to_demand_ratio": total_capacity / total_demand,
            "unit_cost_roundtrip_max_error": max_error,
        },
    )


def summarize(instance: Instance) -> dict[str, Any]:
    """변환 결과 요약 (검증 출력을 위한 보조 함수)."""
    slack = instance.capacity_slack_max
    return {
        "instance_id": instance.instance_id,
        "name": instance.name,
        "n_facilities": instance.n_facilities,
        "n_customers": instance.n_customers,
        "total_demand": instance.total_demand,
        "total_capacity": instance.total_capacity,
        "capacity_ratio": instance.total_capacity / instance.total_demand,
        "capacity_slack_max": slack,
        "capacity_slack_bits": 0 if slack == 0 else math.ceil(math.log2(slack + 1)),
        "objective_upper_bound": instance.objective_upper_bound,
        "num_zero_fixed_cost": int((instance.fixed_cost == 0).sum()),
    }
