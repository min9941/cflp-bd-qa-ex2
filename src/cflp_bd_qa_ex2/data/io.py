"""인스턴스 저장/로드와 schema 검증."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .generator import Instance

INSTANCE_SCHEMA_VERSION = "1.0"

_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "instance_id",
    "name",
    "seed",
    "n_facilities",
    "n_customers",
    "facility_locations",
    "customer_locations",
    "demand",
    "capacity",
    "fixed_cost",
    "transport_cost",
    "metadata",
)


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    """인스턴스를 JSON 직렬화 가능한 dict 로 변환한다."""
    return {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "instance_id": inst.instance_id,
        "name": inst.name,
        "seed": int(inst.seed),
        "n_facilities": int(inst.n_facilities),
        "n_customers": int(inst.n_customers),
        "facility_locations": inst.facility_locations.tolist(),
        "customer_locations": inst.customer_locations.tolist(),
        "demand": inst.demand.astype(int).tolist(),
        "capacity": inst.capacity.astype(int).tolist(),
        "fixed_cost": inst.fixed_cost.tolist(),
        "transport_cost": inst.transport_cost.tolist(),
        "metadata": inst.metadata,
    }


def instance_from_dict(payload: dict[str, Any]) -> Instance:
    """dict 로부터 인스턴스를 복원한다."""
    validate_instance_payload(payload)
    return Instance(
        instance_id=payload["instance_id"],
        name=payload["name"],
        seed=int(payload["seed"]),
        n_facilities=int(payload["n_facilities"]),
        n_customers=int(payload["n_customers"]),
        facility_locations=np.asarray(payload["facility_locations"], dtype=float),
        customer_locations=np.asarray(payload["customer_locations"], dtype=float),
        demand=np.asarray(payload["demand"], dtype=np.int64),
        capacity=np.asarray(payload["capacity"], dtype=np.int64),
        fixed_cost=np.asarray(payload["fixed_cost"], dtype=float),
        transport_cost=np.asarray(payload["transport_cost"], dtype=float),
        metadata=dict(payload.get("metadata", {})),
    )


def validate_instance_payload(payload: dict[str, Any]) -> None:
    """저장된 인스턴스 payload 의 schema 를 검증한다.

    Raises:
        ValueError: 필수 key 누락, 배열 shape 불일치, demand/capacity 양수 위반 시.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"인스턴스 schema 위반: 필수 key 누락 {missing}")

    nf, nc = int(payload["n_facilities"]), int(payload["n_customers"])
    shapes = {
        "facility_locations": (nf, 2),
        "customer_locations": (nc, 2),
        "transport_cost": (nc, nf),
    }
    for key, expected in shapes.items():
        actual = np.asarray(payload[key]).shape
        if actual != expected:
            raise ValueError(f"인스턴스 schema 위반: {key} shape {actual} != {expected}")

    for key, length in (("demand", nc), ("capacity", nf), ("fixed_cost", nf)):
        arr = np.asarray(payload[key])
        if arr.shape != (length,):
            raise ValueError(f"인스턴스 schema 위반: {key} 길이 {arr.shape} != {(length,)}")

    if np.any(np.asarray(payload["demand"]) < 1):
        raise ValueError("인스턴스 schema 위반: demand 는 1 이상이어야 한다")
    if np.any(np.asarray(payload["capacity"]) < 0):
        raise ValueError("인스턴스 schema 위반: capacity 는 음수일 수 없다")
    if np.any(np.asarray(payload["transport_cost"]) < 0):
        raise ValueError("인스턴스 schema 위반: transport_cost 는 음수일 수 없다")

    total_capacity = int(np.asarray(payload["capacity"]).sum())
    expected_total = payload["metadata"].get("capacity_target_total")
    if expected_total is not None and total_capacity != int(expected_total):
        raise ValueError(
            f"인스턴스 schema 위반: 총용량 {total_capacity} != S_target {expected_total}"
        )


def save_instance(inst: Instance, out_dir: str | Path) -> Path:
    """인스턴스를 JSON 으로 저장한다 (기존 파일은 덮어쓴다)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{inst.instance_id}.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(instance_to_dict(inst), fp, ensure_ascii=False)
    return path


def load_instance(path: str | Path) -> Instance:
    """JSON 파일에서 인스턴스를 읽는다."""
    with open(path, "r", encoding="utf-8") as fp:
        return instance_from_dict(json.load(fp))


def load_all_instances(in_dir: str | Path) -> list[Instance]:
    """디렉터리의 모든 인스턴스를 instance_id 순으로 읽는다."""
    return [load_instance(p) for p in sorted(Path(in_dir).glob("*.json"))]
