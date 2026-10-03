"""설정 schema와 검증.

설정은 dict 로 다루되, 필수 key 존재와 값의 유효 범위를 명시적으로 검사한다.
누락된 값을 임의로 채우지 않는다. 누락은 ``ConfigError`` 로 즉시 실패시킨다.
"""

from __future__ import annotations

from typing import Any, Iterable


class ConfigError(ValueError):
    """설정이 유효하지 않을 때 발생하는 예외."""


REQUIRED_TOP_LEVEL: tuple[str, ...] = (
    "data_generation",
    "benders",
    "gurobi",
    "encoding",
    "penalty",
    "sa",
    "qa",
    "embedding",
    "validation",
)

_VALID_GUARD_ACTIONS = {"stop_run", "continue_with_raw_dual"}


def _require(mapping: dict[str, Any], keys: Iterable[str], where: str) -> None:
    """``mapping`` 에 ``keys`` 가 모두 존재하는지 확인한다."""
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"{where}: 필수 설정 key 누락 {missing}")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """전체 설정을 검증하고 그대로 돌려준다.

    Args:
        cfg: 병합이 끝난 설정 dict.

    Returns:
        검증을 통과한 설정 dict (입력과 동일한 객체).

    Raises:
        ConfigError: 필수 key 가 없거나 값이 유효 범위를 벗어난 경우.
    """
    _require(cfg, REQUIRED_TOP_LEVEL, "config")

    dg = cfg["data_generation"]
    _require(dg, ("demand", "capacity", "fixed_cost", "transport_cost"), "data_generation")
    demand = dg["demand"]
    _require(demand, ("mean", "standard_deviation", "minimum"), "data_generation.demand")
    if demand["standard_deviation"] <= 0:
        raise ConfigError("data_generation.demand.standard_deviation 은 양수여야 한다")
    if dg["capacity"]["total_capacity_ratio"] <= 1.0:
        raise ConfigError(
            "data_generation.capacity.total_capacity_ratio 는 1보다 커야 한다. "
            "그렇지 않으면 capacity slack 범위가 비어 있게 된다"
        )

    enc = cfg["encoding"]
    _require(enc, ("theta", "capacity_slack", "cut_slack"), "encoding")
    if int(enc["theta"]["bits"]) < 1:
        raise ConfigError("encoding.theta.bits 는 1 이상이어야 한다")
    if float(enc["capacity_slack"]["delta"]) != 1.0:
        raise ConfigError(
            "encoding.capacity_slack.delta 는 1이어야 한다. "
            "capacity slack 은 근사하지 않는다 (명세 9절)"
        )
    ratio = float(enc["cut_slack"]["delta_ratio_to_theta"])
    if not (0.0 < ratio <= 1.0):
        raise ConfigError("encoding.cut_slack.delta_ratio_to_theta 는 (0, 1] 범위여야 한다")

    if float(cfg["penalty"]["margin"]) <= 1.0:
        raise ConfigError("penalty.margin 은 1보다 커야 한다")

    bd = cfg["benders"]
    _require(bd, ("relative_gap_tolerance", "max_iterations", "dual_minimality_guard"), "benders")
    if bd.get("dual_postprocessing", "none") != "none":
        raise ConfigError(
            "benders.dual_postprocessing 은 'none' 이어야 한다. "
            "파일럿에서는 dual 을 자동 변환하지 않는다"
        )
    guard = bd["dual_minimality_guard"]
    _require(guard, ("enabled", "atol", "rtol", "action_by_trajectory"), "benders.dual_minimality_guard")
    for source, action in guard["action_by_trajectory"].items():
        if action not in _VALID_GUARD_ACTIONS:
            raise ConfigError(
                f"benders.dual_minimality_guard.action_by_trajectory[{source}] "
                f"값이 잘못되었다: {action!r}. 허용값 {sorted(_VALID_GUARD_ACTIONS)}"
            )

    val = cfg["validation"]
    positive_checks = (
        ("benders.max_iterations", bd["max_iterations"]),
        ("benders.relative_gap_tolerance", bd["relative_gap_tolerance"]),
        ("gurobi.time_limit", cfg["gurobi"]["time_limit"]),
        ("gurobi.feasibility_tolerance", cfg["gurobi"]["feasibility_tolerance"]),
        ("gurobi.numerical_audit_tolerance", cfg["gurobi"]["numerical_audit_tolerance"]),
        ("sa.sweeps", cfg["sa"]["sweeps"]),
        ("embedding.timeout", cfg["embedding"]["timeout"]),
        ("embedding.tries", cfg["embedding"]["tries"]),
        ("validation.mp_feasibility_tolerance", val["mp_feasibility_tolerance"]),
        ("validation.exhaustive_max_variables", val["exhaustive_max_variables"]),
        ("validation.miqp_max_variables", val["miqp_max_variables"]),
        ("validation.miqp_tractable_max_variables", val.get("miqp_tractable_max_variables", 50)),
        ("validation.exact_time_limit", val.get("exact_time_limit", 60)),
    )
    for name, value in positive_checks:
        if value is None or float(value) <= 0:
            raise ConfigError(f"{name} 는 양수여야 한다 (현재 {value!r})")

    qa = cfg["qa"]
    if int(qa.get("num_repeats", 1)) < 1:
        raise ConfigError("qa.num_repeats 는 1 이상이어야 한다")
    if int(qa.get("num_reads", 1)) < 1:
        raise ConfigError("qa.num_reads 는 1 이상이어야 한다")
    if int(cfg["sa"].get("num_reads", 1)) < 1:
        raise ConfigError("sa.num_reads 는 1 이상이어야 한다")

    emb = cfg["embedding"]
    _require(emb, ("timeout", "tries", "seeds", "ideal_targets"), "embedding")
    if not emb["seeds"]:
        raise ConfigError("embedding.seeds 는 비어 있을 수 없다")
    if len(set(emb["seeds"])) != len(emb["seeds"]):
        raise ConfigError("embedding.seeds 에 중복이 있다")

    _require(val, ("exhaustive_max_variables", "miqp_max_variables", "mp_feasibility_tolerance"), "validation")

    return cfg


def validate_instance_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """인스턴스 명세 목록을 검증한다.

    Args:
        specs: ``name``/``facilities``/``customers``/``seeds`` 를 갖는 dict 목록.

    Returns:
        검증을 통과한 목록.

    Raises:
        ConfigError: 필수 key 누락, 크기가 1 미만, seed 중복 시.
    """
    if not specs:
        raise ConfigError("instances 목록이 비어 있다")
    seen: set[tuple[str, int]] = set()
    for spec in specs:
        _require(spec, ("name", "facilities", "customers", "seeds"), "instances")
        if spec["facilities"] < 1 or spec["customers"] < 1:
            raise ConfigError(f"instances[{spec['name']}]: 크기는 1 이상이어야 한다")
        for seed in spec["seeds"]:
            key = (spec["name"], int(seed))
            if key in seen:
                raise ConfigError(f"instances: 중복된 (name, seed) = {key}")
            seen.add(key)
    return specs
