"""설정 로더.

``extends`` key 를 통한 단순 상속(깊은 병합)을 지원하고,
설정 전체의 SHA-256 해시를 계산해 provenance 에 기록한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .schema import validate_config, validate_instance_specs


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` 를 ``base`` 위에 깊은 병합한다.

    dict 가 아닌 값은 통째로 교체한다. list 도 교체이며 병합하지 않는다.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML 파일 하나를 읽는다."""
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: YAML 최상위는 mapping 이어야 한다")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """실험 설정을 읽고 ``extends`` 를 해석한 뒤 검증한다.

    Args:
        path: 실험 설정 파일 경로 (예: ``config/experiments/pilot.yaml``).

    Returns:
        병합·검증이 끝난 설정 dict. ``instances`` 가 있으면 함께 검증된다.
    """
    path = Path(path)
    cfg = load_yaml(path)
    parent = cfg.pop("extends", None)
    if parent is not None:
        base = load_config(( path.parent / parent).resolve())
        base.pop("configuration_hash", None)
        cfg = _deep_merge(base, cfg)

    if "instances" in cfg:
        validate_instance_specs(cfg["instances"])
    # 최상위 설정(base.yaml 자체 로드)에서는 instances 가 없을 수 있다.
    if all(key in cfg for key in ("data_generation", "benders", "encoding")):
        validate_config(cfg)
    cfg["configuration_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """설정의 결정적 SHA-256 해시를 계산한다.

    ``configuration_hash`` key 자체는 해시 계산에서 제외한다.
    """
    payload = {k: v for k, v in cfg.items() if k != "configuration_hash"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def snapshot_config(cfg: dict[str, Any], out_dir: str | Path) -> Path:
    """설정의 완전한 snapshot 을 결과 폴더에 저장한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"config_snapshot_{cfg.get('configuration_hash', 'nohash')[:12]}.yaml"
    with open(target, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=True)
    return target
