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


def _nullable(values: np.ndarray) -> list:
    """NaN 을 ``None`` 으로 바꿔 JSON 에 ``null`` 로 기록되게 한다.

    ``json.dump`` 는 기본적으로 ``NaN`` 을 그대로 쓰는데, 이는 엄격한 JSON 이
    아니라 다른 도구에서 읽지 못할 수 있다. 벤치마크 인스턴스는 좌표가 없어
    ``NaN`` 이 들어가므로 변환이 필요하다.
    """
    return [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in np.atleast_2d(values)
    ]


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    """인스턴스를 JSON 직렬화 가능한 dict 로 변환한다."""
    return {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "instance_id": inst.instance_id,
        "name": inst.name,
        "seed": int(inst.seed),
        "n_facilities": int(inst.n_facilities),
        "n_customers": int(inst.n_customers),
        "facility_locations": _nullable(inst.facility_locations),
        "customer_locations": _nullable(inst.customer_locations),
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
        # 좌표는 NaN(벤치마크) 일 수 있으므로 shape 만 검사한다.
        actual = np.asarray(payload[key], dtype=float).shape
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


def load_instances(*dirs: str | Path, required: bool = False) -> list[Instance]:
    """여러 디렉터리의 인스턴스를 모아 읽는다.

    생성 데이터(`data/raw`)와 벤치마크(`data/benchmark`)를 함께 쓰기 위한 함수이다.
    존재하지 않는 디렉터리는 조용히 건너뛴다(벤치마크를 아직 변환하지 않았을 수 있다).

    Args:
        *dirs: 인스턴스 JSON 이 있는 디렉터리들.
        required: True 면 존재하지 않는 디렉터리에서 예외를 낸다.

    Returns:
        ``instance_id`` 순으로 정렬된 인스턴스 목록.

    Raises:
        ValueError: 서로 다른 디렉터리에 같은 ``instance_id`` 가 있는 경우.
            어느 쪽을 쓸지 임의로 정하지 않는다.
        FileNotFoundError: ``required=True`` 인데 디렉터리가 없는 경우.
    """
    collected: dict[str, tuple[Instance, Path]] = {}
    for directory in dirs:
        path = Path(directory)
        if not path.is_dir():
            if required:
                raise FileNotFoundError(f"인스턴스 디렉터리가 없다: {path}")
            continue
        for instance in load_all_instances(path):
            if instance.instance_id in collected:
                previous = collected[instance.instance_id][1]
                raise ValueError(
                    f"중복된 instance_id '{instance.instance_id}': "
                    f"{previous} 와 {path} 에 모두 있다. 어느 쪽을 쓸지 정해야 한다"
                )
            collected[instance.instance_id] = (instance, path)
    return [item[0] for _, item in sorted(collected.items())]
