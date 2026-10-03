"""MS-CFLP 인스턴스 생성기.

Cornuejols 계열 생성법(capacity ~ U(10,160), f_j = U(0,90) + U(100,110)*sqrt(s_j),
c_ij = 10*dist, 총용량비 1.5)을 따르되 **demand 분포만 변경**한 변형이다.
원래 생성법의 demand 는 U[5,35] 로 설명되는 반면 본 명세는 N(35, 5^2) 이다.

draw 순서는 재현성을 위해 아래로 **고정**한다. 순서를 바꾸면 같은 seed 라도
다른 인스턴스가 나온다.

1. facility_locations : rng.random((n_facilities, 2))
2. customer_locations : rng.random((n_customers, 2))
3. demand_raw         : rng.normal(mean, standard_deviation, n_customers)
4. capacity_raw       : rng.uniform(raw_low, raw_high, n_facilities)
5. fixed_cost_base    : rng.uniform(base_low, base_high, n_facilities)
6. fixed_cost_slope   : rng.uniform(slope_low, slope_high, n_facilities)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DRAW_ORDER: tuple[str, ...] = (
    "facility_locations",
    "customer_locations",
    "demand_raw",
    "capacity_raw",
    "fixed_cost_base",
    "fixed_cost_slope",
)


@dataclass(frozen=True)
class Instance:
    """하나의 MS-CFLP 인스턴스.

    Attributes:
        instance_id: ``{name}_s{seed}`` 형식의 식별자.
        name: 크기 label (예: ``"4x12"``).
        seed: 생성에 쓰인 seed.
        n_facilities: 시설 수 (첨자 j).
        n_customers: 고객 수 (첨자 i).
        facility_locations: shape ``(n_facilities, 2)``.
        customer_locations: shape ``(n_customers, 2)``.
        demand: 정수 demand, shape ``(n_customers,)``.
        capacity: 정수 capacity, shape ``(n_facilities,)``.
        fixed_cost: 실수 fixed cost, shape ``(n_facilities,)``.
        transport_cost: ``c_ij``, shape ``(n_customers, n_facilities)``.
    """

    instance_id: str
    name: str
    seed: int
    n_facilities: int
    n_customers: int
    facility_locations: np.ndarray
    customer_locations: np.ndarray
    demand: np.ndarray
    capacity: np.ndarray
    fixed_cost: np.ndarray
    transport_cost: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_demand(self) -> int:
        """총수요 D = sum_i d_i."""
        return int(self.demand.sum())

    @property
    def total_capacity(self) -> int:
        """총용량 sum_j s_j."""
        return int(self.capacity.sum())

    @property
    def objective_upper_bound(self) -> float:
        """U_obj = sum_j f_j + sum_i sum_j c_ij d_i.

        theta 의 상한으로 사용한다. 각 고객이 시설 하나만 이용하는데도
        모든 j 에 대해 합산하므로 매우 느슨한 상한이다.
        """
        return float(self.fixed_cost.sum() + (self.transport_cost * self.demand[:, None]).sum())

    @property
    def capacity_slack_max(self) -> int:
        """Capacity slack 의 최대값 S_cap_max = sum_j s_j - D."""
        return self.total_capacity - self.total_demand


def _integerize_capacity(raw: np.ndarray, target_total: int) -> np.ndarray:
    """raw capacity 를 비례 rescale 한 뒤 정수화한다.

    명세 5절의 규칙을 그대로 따른다.

    1. rescale 후 floor
    2. 부족한 unit 을 fractional remainder 가 큰 시설부터 배분
    3. remainder 가 같으면 facility index 가 작은 시설부터 배분

    Args:
        raw: rescale 전 capacity.
        target_total: 정수화 후 반드시 일치해야 하는 총합.

    Returns:
        합이 정확히 ``target_total`` 인 정수 capacity 배열.
    """
    scaled = raw / raw.sum() * target_total
    floored = np.floor(scaled).astype(np.int64)
    deficit = int(target_total - floored.sum())
    if deficit < 0:  # pragma: no cover - floor 특성상 발생하지 않음
        raise RuntimeError("capacity 정수화에서 floor 합이 target 을 초과했다")

    remainder = scaled - floored
    # remainder 내림차순, 동률이면 index 오름차순
    order = sorted(range(len(raw)), key=lambda j: (-remainder[j], j))
    for k in range(deficit):
        floored[order[k % len(order)]] += 1

    if int(floored.sum()) != target_total:  # pragma: no cover - 방어적
        raise RuntimeError("capacity 정수화 후 총합이 target 과 다르다")
    return floored


def generate_instance(
    name: str,
    n_facilities: int,
    n_customers: int,
    seed: int,
    cfg: dict[str, Any],
) -> Instance:
    """설정과 seed 로부터 인스턴스 하나를 결정적으로 생성한다.

    Args:
        name: 크기 label (예: ``"8x25"``).
        n_facilities: 시설 수.
        n_customers: 고객 수.
        seed: ``PCG64`` seed.
        cfg: 전체 설정 dict (``data_generation`` 하위를 사용).

    Returns:
        생성된 :class:`Instance`.

    Raises:
        RuntimeError: 총용량 equality assert 가 깨진 경우.
    """
    dg = cfg["data_generation"]
    rng = np.random.Generator(np.random.PCG64(seed))

    # --- draw 순서 고정 (DRAW_ORDER 와 반드시 일치) ---
    facility_locations = rng.random((n_facilities, 2))
    customer_locations = rng.random((n_customers, 2))

    dcfg = dg["demand"]
    demand_raw = rng.normal(
        loc=float(dcfg["mean"]),
        scale=float(dcfg["standard_deviation"]),
        size=n_customers,
    )

    ccfg = dg["capacity"]
    capacity_raw = rng.uniform(float(ccfg["raw_low"]), float(ccfg["raw_high"]), n_facilities)

    fcfg = dg["fixed_cost"]
    fixed_base = rng.uniform(float(fcfg["base_low"]), float(fcfg["base_high"]), n_facilities)
    fixed_slope = rng.uniform(float(fcfg["slope_low"]), float(fcfg["slope_high"]), n_facilities)
    # --- draw 종료 ---

    minimum = int(dcfg["minimum"])
    demand = np.maximum(minimum, np.rint(demand_raw)).astype(np.int64)

    total_demand = int(demand.sum())
    target_total = math.ceil(float(ccfg["total_capacity_ratio"]) * total_demand)
    capacity = _integerize_capacity(capacity_raw, target_total)

    if int(capacity.sum()) != target_total:
        raise RuntimeError(
            f"총용량 equality 위반: sum(s_j)={capacity.sum()} != S_target={target_total}"
        )

    fixed_cost = fixed_base + fixed_slope * np.sqrt(capacity.astype(float))

    diff = customer_locations[:, None, :] - facility_locations[None, :, :]
    distance = np.linalg.norm(diff, axis=2)
    transport_cost = float(dg["transport_cost"]["distance_multiplier"]) * distance

    return Instance(
        instance_id=f"{name}_s{seed}",
        name=name,
        seed=seed,
        n_facilities=n_facilities,
        n_customers=n_customers,
        facility_locations=facility_locations,
        customer_locations=customer_locations,
        demand=demand,
        capacity=capacity,
        fixed_cost=fixed_cost,
        transport_cost=transport_cost,
        metadata={
            "draw_order": list(DRAW_ORDER),
            "total_demand": total_demand,
            "capacity_target_total": target_total,
            "demand_distribution": "normal(mean=35, standard_deviation=5)",
            "generator_family": "Cornuejols-style with modified demand distribution",
        },
    )


def generate_all(cfg: dict[str, Any]) -> list[Instance]:
    """설정의 ``instances`` 명세 전체를 생성한다."""
    out: list[Instance] = []
    for spec in cfg["instances"]:
        for seed in spec["seeds"]:
            out.append(
                generate_instance(
                    name=spec["name"],
                    n_facilities=int(spec["facilities"]),
                    n_customers=int(spec["customers"]),
                    seed=int(seed),
                    cfg=cfg,
                )
            )
    return out
