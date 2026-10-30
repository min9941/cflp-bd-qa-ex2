"""Theta bound 전략의 offline 분석.

기존에 저장된 cut trajectory 를 재사용해, SA 나 QPU 를 다시 돌리지 않고
bound 전략과 bit 수가 QUBO 구조에 주는 영향을 계산한다.

**이 모듈은 자원과 정밀도만 다룬다.** raw feasibility, solution gap,
embedding time 은 별도 실험이 필요하며 여기서 주장할 수 없다.

trajectory 종류마다 cut 이 다르므로 **절대 섞지 않는다.**
Gurobi 의 25번째 iteration 과 QA 의 25번째 iteration 은 다른 cut 집합이다.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.generator import Instance
from ..encoding.bounds import ThetaBound
from ..encoding.pure_exponential import bits_for_range
from ..encoding.slack import CutSlackRangeInvalid
from ..encoding.theta import bits_for_target_resolution

TRAJECTORY_SOURCES: tuple[str, ...] = ("gurobi", "sa", "pegasus", "zephyr")


@dataclass(frozen=True)
class ObservedIteration:
    """실제로 실행된 MP의 iteration 번호와 시작 시점 cut 수."""

    iteration: int
    num_cuts: int


@dataclass(frozen=True)
class ObservedTrajectory:
    """Cut과 원본 실행 시점을 함께 보존한 분석 입력."""

    cuts: tuple[CutRecord, ...]
    observations: tuple[ObservedIteration, ...]
    trajectory_hash: str

    @property
    def last_iteration(self) -> int:
        """Dual 유무와 무관한 실제 마지막 실행 번호."""
        return self.observations[-1].iteration


@dataclass(frozen=True)
class CutRecord:
    """한 iteration 에서 생성된 Benders cut 의 dual 정보."""

    iteration: int
    dual_u: np.ndarray
    dual_v: np.ndarray
    activated: bool

    @property
    def u_sum(self) -> float:
        """sum_i u_i."""
        return float(self.dual_u.sum())


def _parse_vector(value: Any) -> np.ndarray | None:
    """CSV 에 문자열로 저장된 리스트를 배열로 되돌린다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    try:
        return np.asarray(ast.literal_eval(str(value)), dtype=float)
    except (ValueError, SyntaxError):
        return None


def load_cut_trajectory(
    results_root: str | Path,
    source: str,
    instance_id: str,
    configuration_hash: str | None = None,
    graph_id: str | None = None,
) -> list[CutRecord]:
    """저장된 결과에서 한 trajectory 의 cut 목록을 복원한다.

    - ``gurobi``: ``results/benders/cut_trajectory.csv`` 에 dual 과
      ``cut_activated`` 가 그대로 있다.
    - ``sa`` / ``pegasus`` / ``zephyr``: iteration 기록의 ``raw_dual_u`` /
      ``raw_dual_v`` 로 cut 을 복원하고, ``num_benders_cuts`` 의 증가로
      활성화 여부를 판정한다. (iteration k 의 cut 이 활성화되면
      iteration k+1 의 활성 cut 수가 1 늘어난다.)

    Args:
        results_root: ``results`` 디렉터리.
        source: ``"gurobi"``, ``"sa"``, ``"pegasus"``, ``"zephyr"``.
        instance_id: 대상 인스턴스.
        configuration_hash: 주면 그 설정의 실행만 쓴다.
        graph_id: 주면 그 working graph 의 실행만 쓴다 (QA 전용).

    Returns:
        iteration 오름차순 :class:`CutRecord` 목록.

    Raises:
        FileNotFoundError: 해당 결과 파일이 없는 경우.
        ValueError: 알 수 없는 source 이거나, 같은 instance 에 서로 다른
            실행이 섞여 있는데 식별자로 좁혀지지 않은 경우. 두 trajectory 를
            이어 붙여 activation 을 계산하면 결과가 조용히 틀어진다.
    """

    def _narrow(frame: pd.DataFrame) -> pd.DataFrame:
        """식별자로 좁히고, 그래도 섞여 있으면 거부한다."""
        if configuration_hash and "configuration_hash" in frame.columns:
            frame = frame[frame["configuration_hash"] == configuration_hash]
        if graph_id and "graph_id" in frame.columns:
            frame = frame[frame["graph_id"] == graph_id]
        if frame.empty:
            return frame
        for column in ("configuration_hash", "graph_id"):
            if column in frame.columns:
                distinct = frame[column].dropna().unique()
                if len(distinct) > 1:
                    raise ValueError(
                        f"{instance_id}/{source}: {column} 가 {len(distinct)}종 섞여 있다. "
                        f"{column} 를 지정해 하나로 좁혀야 한다 ({list(distinct)[:3]})"
                    )
        duplicated = frame["benders_iteration"].duplicated().sum()
        if duplicated:
            raise ValueError(
                f"{instance_id}/{source}: benders_iteration 이 {duplicated}건 중복된다. "
                "서로 다른 실행이 섞여 있다"
            )
        return frame

    results_root = Path(results_root)
    if source not in TRAJECTORY_SOURCES:
        raise ValueError(f"알 수 없는 trajectory source: {source!r}")

    if source == "gurobi":
        path = results_root / "benders" / "cut_trajectory.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없다")
        frame = pd.read_csv(path)
        frame = frame[frame["instance_id"] == instance_id].sort_values("benders_iteration")
        frame = _narrow(frame)
        records = []
        for row in frame.itertuples():
            u = _parse_vector(row.cut_dual_u)
            v = _parse_vector(row.cut_dual_v)
            if u is None or v is None:
                continue
            records.append(
                CutRecord(
                    iteration=int(row.benders_iteration),
                    dual_u=u,
                    dual_v=v,
                    activated=bool(getattr(row, "cut_activated", True)),
                )
            )
        return records

    path = (
        results_root / "sa" / "sa_iterations.csv"
        if source == "sa"
        else results_root / "qa" / f"qa_iterations_{source}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없다")

    frame = pd.read_csv(path)
    frame = frame[frame["instance_id"] == instance_id].sort_values("benders_iteration")
    frame = _narrow(frame)
    if frame.empty:
        return []

    active_counts = frame["num_benders_cuts"].tolist()
    records = []
    for position, row in enumerate(frame.itertuples()):
        u = _parse_vector(row.raw_dual_u)
        v = _parse_vector(row.raw_dual_v)
        if u is None or v is None:
            continue
        # 다음 iteration 의 활성 cut 수가 늘었으면 이 cut 이 추가된 것이다.
        if position + 1 < len(active_counts):
            activated = active_counts[position + 1] > active_counts[position]
        else:
            activated = False  # 마지막 iteration 의 cut 은 쓰인 적이 없다
        records.append(
            CutRecord(
                iteration=int(row.benders_iteration),
                dual_u=u,
                dual_v=v,
                activated=bool(activated),
            )
        )
    return records


def load_observed_trajectory(
    results_root: str | Path,
    source: str,
    instance_id: str,
    configuration_hash: str | None = None,
    graph_id: str | None = None,
) -> ObservedTrajectory:
    """원본 iteration 행을 기준으로 cut 복원의 완전성과 실행 시점을 검사한다.

    마지막 embedding 실패 행에는 dual이 없어도 실행된 MP가 있으므로 반드시
    보존한다. Cut 생성 직후 상태는 해당 iteration의 실행 MP에 포함하지 않는다.
    """
    root = Path(results_root)
    if source not in TRAJECTORY_SOURCES:
        raise ValueError(f"알 수 없는 trajectory source: {source!r}")
    relative = (
        "benders/benders_iterations.csv" if source == "gurobi" else
        "sa/sa_iterations.csv" if source == "sa" else
        f"qa/qa_iterations_{source}.csv"
    )
    frame = pd.read_csv(root / relative)
    frame = frame[frame["instance_id"] == instance_id].copy()
    for column, selected in (("configuration_hash", configuration_hash), ("graph_id", graph_id)):
        if selected is not None:
            if column not in frame:
                raise ValueError(f"{instance_id}/{source}: {column} 열이 없다")
            frame = frame[frame[column] == selected]
    if frame.empty:
        raise ValueError(f"{instance_id}/{source}: 선택한 실행의 iteration 기록이 없다")
    for column in ("configuration_hash", "graph_id", "trajectory_source"):
        if column in frame and frame[column].nunique(dropna=False) > 1:
            raise ValueError(f"{instance_id}/{source}: {column}가 혼합되어 있다")
    frame = frame.sort_values("benders_iteration").reset_index(drop=True)
    numbers = pd.to_numeric(frame["benders_iteration"], errors="raise").to_numpy(float)
    counts = pd.to_numeric(frame["num_benders_cuts"], errors="raise").to_numpy(float)
    if not np.array_equal(numbers, np.arange(1, len(frame) + 1)):
        raise ValueError(f"{instance_id}/{source}: iteration 중복 또는 누락이 있다")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError(f"{instance_id}/{source}: num_benders_cuts가 유효하지 않다")
    selected_config = (
        str(frame["configuration_hash"].iloc[0])
        if "configuration_hash" in frame and pd.notna(frame["configuration_hash"].iloc[0])
        else configuration_hash
    )
    selected_graph = (
        str(frame["graph_id"].iloc[0])
        if "graph_id" in frame and pd.notna(frame["graph_id"].iloc[0]) else graph_id
    )
    cuts = load_cut_trajectory(root, source, instance_id, selected_config, selected_graph)
    for cut in cuts:
        if (cut.dual_u.ndim != 1 or cut.dual_v.ndim != 1
                or not np.all(np.isfinite(cut.dual_u)) or not np.all(np.isfinite(cut.dual_v))):
            raise ValueError(f"{instance_id}/{source}: iteration {cut.iteration}의 dual이 유효하지 않다")
    active = sorted((c for c in cuts if c.activated), key=lambda c: c.iteration)
    pointer = 0
    observations = []
    for iteration, expected in zip(numbers.astype(int), counts.astype(int)):
        while pointer < len(active) and active[pointer].iteration < iteration:
            pointer += 1
        if pointer != expected:
            raise ValueError(
                f"{instance_id}/{source}: iteration {iteration} cut 복원 불일치 "
                f"(복원 {pointer}, 원본 {expected})"
            )
        observations.append(ObservedIteration(int(iteration), int(expected)))
    # 선택한 입력 내용 자체를 식별한다. Gurobi는 별도 cut 파일도 포함한다.
    payload = frame.to_json(orient="split", double_precision=15, force_ascii=False)
    payload += json.dumps([
        {"iteration": c.iteration, "u": c.dual_u.tolist(), "v": c.dual_v.tolist(),
         "activated": c.activated} for c in cuts
    ], sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(f"{source}/{instance_id}/{payload}".encode("utf-8")).hexdigest()
    return ObservedTrajectory(tuple(cuts), tuple(observations), digest)


def cut_slack_bits(
    cut: CutRecord,
    capacity: np.ndarray,
    theta_upper_bound: float,
    delta_cut: float,
    negative_tolerance: float = 1e-6,
) -> tuple[int, float, float]:
    """cut 하나의 slack bit 수를 계산한다.

    ``S_k_max = max(0, theta_U + sum_j s_j v_j - sum_i u_i)`` 이고
    ``b_k = ceil(log2(S_k_max / delta_cut + 1))`` 이다.
    ``S_k_max`` 는 theta_L 과 무관하지만 ``delta_cut`` 은 theta_L 에 의존한다.

    Returns:
        ``(bits, S_k_max, A_k)``. ``A_k = sum_i u_i - sum_j s_j v_j`` 는
        cut 을 ``y = 1`` 에서 평가한 값이다.

    Raises:
        CutSlackRangeInvalid: ``S_k_max`` 가 허용 오차를 넘어 음수인 경우.
            실제 builder 와 동일한 규칙이다.
    """
    a_k = cut.u_sum - float(capacity.astype(float) @ cut.dual_v)
    raw_max = float(theta_upper_bound) - a_k

    # 실제 builder(build_cut_slack_encoding)와 **같은 규칙**을 쓴다.
    # offline 분석만 조용히 0 으로 clamp 하면 같은 입력에 다른 결과가 나오고,
    # 명세 2절의 중단 규칙과도 어긋난다.
    if raw_max < -negative_tolerance:
        raise CutSlackRangeInvalid(
            f"iteration {cut.iteration}: S_k_max = {raw_max!r} < -{negative_tolerance} "
            "(CUT_SLACK_RANGE_INVALID)"
        )
    clamped = raw_max < 0.0
    if clamped:
        raw_max = 0.0  # 허용 오차 안의 수치 오차만 0 으로 본다

    bits = bits_for_range(raw_max, delta_cut)
    return bits, raw_max, a_k


def analyze_configuration(
    inst: Instance,
    bound: ThetaBound,
    cuts: Sequence[CutRecord],
    *,
    theta_bits: int | None = None,
    target_delta: float | None = None,
    cut_slack_ratio: float = 0.25,
    capacity_slack_bits: int | None = None,
    compute_edges: bool = True,
    negative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """(bound 전략, bit 설정) 하나에 대한 QUBO 구조 지표를 계산한다.

    ``theta_bits`` 와 ``target_delta`` 중 **정확히 하나**를 준다.

    - ``theta_bits`` 고정 → E1-A (Fixed Bit Budget) / E2 (Bit Sweep)
    - ``target_delta`` 고정 → E1-B (Fixed Target Resolution).
      실제 ``delta_theta`` 는 목표값 **이하**이며 같지 않다.

    Args:
        inst: 대상 인스턴스.
        bound: theta 구간.
        cuts: QUBO 에 들어갈 활성 cut 목록.
        theta_bits: 고정할 bit 수.
        target_delta: 넘지 않아야 할 목표 해상도.
        cut_slack_ratio: ``delta_cut = ratio * delta_theta`` 의 ratio.
        capacity_slack_bits: 미리 계산한 capacity slack bit 수.
        compute_edges: False 면 edge 집합을 만들지 않는다. 전 구간 분석처럼
            호출이 많을 때 쓴다. edge 관련 값은 ``None`` 이 된다.
        negative_tolerance: 실제 builder와 동일하게 사용할 slack 범위 허용 오차.

    Returns:
        bit 수, 해상도, 변수/엣지 추정치를 담은 dict.
    """
    if (theta_bits is None) == (target_delta is None):
        raise ValueError("theta_bits 와 target_delta 중 정확히 하나를 지정해야 한다")

    span = bound.theta_range
    if not math.isfinite(span) or span <= 0 or not math.isfinite(cut_slack_ratio) or cut_slack_ratio <= 0:
        raise ValueError("theta 범위와 cut slack ratio는 유한한 양수여야 한다")
    if theta_bits is None:
        bits = bits_for_target_resolution(
            bound.theta_lower_bound, bound.theta_upper_bound, float(target_delta)
        )
        bit_mode = "fixed_target_resolution"
    else:
        bits = int(theta_bits)
        if bits < 1 or bits != theta_bits:
            raise ValueError("theta_bits는 1 이상의 정수여야 한다")
        bit_mode = "fixed_bit_budget"

    delta_theta = span / (2**bits - 1)
    delta_cut = cut_slack_ratio * delta_theta

    if capacity_slack_bits is None:
        slack_max = inst.total_capacity - inst.total_demand
        capacity_slack_bits = bits_for_range(float(slack_max), 1.0)

    per_cut_bits: list[int] = []
    a_values: list[float] = []
    for cut in cuts:
        bits_k, _, a_k = cut_slack_bits(
            cut, inst.capacity.astype(float), bound.theta_upper_bound, delta_cut,
            negative_tolerance=negative_tolerance,
        )
        per_cut_bits.append(bits_k)
        a_values.append(a_k)

    cut_slack_total = int(sum(per_cut_bits))
    logical_variables = inst.n_facilities + bits + capacity_slack_bits + cut_slack_total

    # logical edges: 각 equality residual 을 제곱하면 그 안의 변수들이 clique 를
    # 이룬다. 여러 constraint 가 y-y, y-theta, theta-theta edge 를 **공유**하므로
    # clique 크기를 단순 합산하면 중복 계산이 된다. 고유 edge 의 합집합을 센다.
    # 계수가 0 인 변수(예: v_j = 0 인 시설)는 builder 가 항을 버리므로 제외한다.
    if not compute_edges:
        logical_edges = None
        interaction_upper_bound = None
    edge_set: set[tuple[int, int]] = set()
    interaction_upper_bound = 0 if compute_edges else None

    def _add_clique(members: list[int]) -> None:
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                lo, hi = sorted((members[a_idx], members[b_idx]))
                edge_set.add((lo, hi))

    # 변수에 정수 id 를 부여한다: y, theta, capacity slack, cut slack 순
    y_ids = list(range(inst.n_facilities))
    theta_ids = list(range(inst.n_facilities, inst.n_facilities + bits))
    cap_ids = list(
        range(inst.n_facilities + bits, inst.n_facilities + bits + capacity_slack_bits)
    )
    next_id = inst.n_facilities + bits + capacity_slack_bits

    if compute_edges:
        capacity_members = [j for j in y_ids if inst.capacity[j] != 0] + cap_ids
        _add_clique(capacity_members)
        interaction_upper_bound += len(capacity_members) * (len(capacity_members) - 1) // 2

        for cut, bits_k in zip(cuts, per_cut_bits):
            slack_ids = list(range(next_id, next_id + bits_k))
            next_id += bits_k
            active_y = [j for j in y_ids if cut.dual_v[j] != 0.0]
            members = active_y + theta_ids + slack_ids
            _add_clique(members)
            interaction_upper_bound += len(members) * (len(members) - 1) // 2
        logical_edges = len(edge_set)
    else:
        logical_edges = None

    return {
        "instance_id": inst.instance_id,
        "bound_strategy": bound.strategy_id,
        "bound_strategy_name": bound.strategy_name,
        "bit_mode": bit_mode,
        "theta_lower_bound": bound.theta_lower_bound,
        "theta_upper_bound": bound.theta_upper_bound,
        "theta_range": span,
        "theta_bits": bits,
        "target_delta_theta": target_delta,
        "delta_theta": delta_theta,
        "delta_cut": delta_cut,
        "theta_relative_resolution": delta_theta / span if span else float("nan"),
        "num_cuts": len(cuts),
        "capacity_slack_bits": capacity_slack_bits,
        "cut_slack_bits_total": cut_slack_total,
        "cut_slack_bits_min": min(per_cut_bits) if per_cut_bits else None,
        "cut_slack_bits_mean": float(np.mean(per_cut_bits)) if per_cut_bits else None,
        "cut_slack_bits_max": max(per_cut_bits) if per_cut_bits else None,
        "logical_variables": logical_variables,
        "logical_edges": logical_edges,
        "constraint_interaction_upper_bound": interaction_upper_bound,
        "num_cuts_with_negative_a": int(sum(1 for a in a_values if a < 0)),
        "a_min": float(min(a_values)) if a_values else None,
        "a_max": float(max(a_values)) if a_values else None,
    }


def active_cuts_at(cuts: Sequence[CutRecord], iteration: int) -> list[CutRecord]:
    """iteration 시점에 QUBO 에 들어 있던 활성 cut 목록."""
    return [c for c in cuts if c.activated and c.iteration < iteration]


def iteration_snapshots(cuts: Sequence[CutRecord], limit: int | None = None) -> list[int]:
    """분석할 iteration 번호 목록. ``limit`` 을 주면 거기까지만."""
    if not cuts:
        return [0]
    last = max(c.iteration for c in cuts)
    if limit is not None:
        last = min(last, limit)
    return list(range(0, last + 1))


def analyze_trajectory(
    inst: Instance,
    bound: ThetaBound,
    cuts: Sequence[CutRecord],
    *,
    theta_bits: int | None = None,
    target_delta: float | None = None,
    cut_slack_ratio: float = 0.25,
    observations: Sequence[ObservedIteration] | None = None,
    negative_tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """iteration 별 자원 증가를 계산한다 (전 구간 분석).

    cut 하나의 slack bit 수는 iteration 과 무관하게 (bound, bit 설정)만으로
    정해진다. 따라서 cut 별 bit 를 **한 번만** 계산하고 누적합으로
    모든 iteration 을 얻는다. iteration 마다 다시 세면 O(n^2) 이 된다.

    edge 는 계산하지 않는다. 정확한 edge 는 대표 snapshot 에서만 구한다.

    Args:
        inst: 대상 인스턴스.
        bound: theta 구간.
        cuts: 활성 cut 목록 (iteration 오름차순).
        theta_bits: 고정할 bit 수 (E1-A / E2).
        target_delta: 목표 해상도 (E1-B).
        cut_slack_ratio: ``delta_cut = ratio * delta_theta``.
        observations: 원본 iteration CSV의 실행 번호와 시작 시점 cut 수.
        negative_tolerance: 실제 builder와 동일한 slack 범위 허용 오차.

    Returns:
        원본에서 관측한 각 iteration의 실행 MP 행. Cut 미추가 iteration과
        마지막 실패 iteration도 포함하며, 인위적인 iteration 0은 만들지 않는다.
    """
    if observations is None or not observations:
        raise ValueError("원본 CSV의 observations를 제공해야 실행 MP를 복원할 수 있다")
    if [o.iteration for o in observations] != sorted({o.iteration for o in observations}):
        raise ValueError("observations는 중복 없이 iteration 오름차순이어야 한다")
    ordered = sorted((c for c in cuts if c.activated), key=lambda c: c.iteration)
    base = analyze_configuration(
        inst, bound, [],
        theta_bits=theta_bits, target_delta=target_delta,
        cut_slack_ratio=cut_slack_ratio, compute_edges=False,
        negative_tolerance=negative_tolerance,
    )
    bits = base["theta_bits"]
    delta_cut = base["delta_cut"]
    capacity_bits = base["capacity_slack_bits"]
    fixed_part = inst.n_facilities + bits + capacity_bits

    common = {k: base[k] for k in (
            "instance_id", "bound_strategy", "bound_strategy_name", "bit_mode",
            "theta_lower_bound", "theta_upper_bound", "theta_range", "theta_bits",
            "target_delta_theta", "delta_theta", "delta_cut",
            "theta_relative_resolution", "capacity_slack_bits",
        )}
    rows = []

    running_bits = 0
    running_negative = 0
    pointer = 0
    for observation in observations:
        while pointer < len(ordered) and ordered[pointer].iteration < observation.iteration:
            bits_k, _, a_k = cut_slack_bits(
                ordered[pointer], inst.capacity.astype(float), bound.theta_upper_bound,
                delta_cut, negative_tolerance=negative_tolerance,
            )
            running_bits += bits_k
            running_negative += int(a_k < 0)
            pointer += 1
        if pointer != observation.num_cuts:
            raise ValueError(
                f"iteration {observation.iteration}: cut 복원 {pointer}개, "
                f"원본 {observation.num_cuts}개로 다르다"
            )
        rows.append({
            **common,
            "benders_iteration": observation.iteration,
            "num_cuts": pointer,
            "cut_slack_bits_total": running_bits,
            "logical_variables": fixed_part + running_bits,
            "num_cuts_with_negative_a": running_negative,
        })
    return rows


def verify_snapshot_qubo(
    inst: Instance,
    bound: ThetaBound,
    cuts: Sequence[CutRecord],
    cfg: dict,
    *,
    theta_bits: int,
    cut_slack_ratio: float = 0.25,
) -> dict[str, Any]:
    """대표 snapshot 에서 **실제 QUBO** 를 만들어 수식 분석과 대조한다.

    수식 분석은 변수 수와 edge 수를 예측할 뿐 실제 계수 범위는 모른다.
    여기서 실제 QUBO 를 만들어 둘을 비교하고 coefficient range 를 측정한다.

    Returns:
        수식값, 실제값, 차이를 담은 dict.
    """
    from ..benders.cuts import build_optimality_cut
    from ..qubo.builder import build_master_qubo
    from ..qubo.diagnostics import qubo_metrics

    ordered = sorted(cuts, key=lambda c: c.iteration)
    real_cuts = [
        build_optimality_cut(
            index, inst.capacity.astype(float), cut.dual_u, cut.dual_v, cut.iteration
        )
        for index, cut in enumerate(ordered)
    ]
    build_cfg = copy.deepcopy(cfg)
    build_cfg["encoding"]["cut_slack"]["delta_ratio_to_theta"] = cut_slack_ratio
    negative_tolerance = float(build_cfg["encoding"]["cut_slack"].get("negative_tolerance", 1e-6))
    qubo = build_master_qubo(
        inst, real_cuts, build_cfg,
        theta_lower_bound=bound.theta_lower_bound,
        theta_upper_bound=bound.theta_upper_bound,
        theta_bits=theta_bits,
    )
    metrics = qubo_metrics(qubo)
    analytic = analyze_configuration(
        inst, bound, ordered, theta_bits=theta_bits,
        cut_slack_ratio=cut_slack_ratio, compute_edges=True,
        negative_tolerance=negative_tolerance,
    )

    return {
        "instance_id": inst.instance_id,
        "bound_strategy": bound.strategy_id,
        "theta_bits": theta_bits,
        "num_cuts": len(ordered),
        "analytic_variables": analytic["logical_variables"],
        "actual_variables": metrics["qubo_num_variables"],
        "variables_match": analytic["logical_variables"] == metrics["qubo_num_variables"],
        "analytic_edges": analytic["logical_edges"],
        "actual_edges": metrics["qubo_num_quadratic_terms"],
        "edges_match": analytic["logical_edges"] == metrics["qubo_num_quadratic_terms"],
        "constraint_interaction_upper_bound": analytic["constraint_interaction_upper_bound"],
        "qubo_density": metrics["qubo_density"],
        "qubo_coefficient_range": metrics["qubo_coefficient_range"],
        "qubo_log10_coefficient_range": metrics["qubo_log10_coefficient_range"],
        "qubo_offset": metrics["qubo_offset"],
    }


def analysis_config_hash(
    *,
    cut_slack_ratio: float,
    reference_bits: int,
    bit_sweep: Sequence[int],
    snapshot_policy: str,
    snapshot_iterations: Sequence[int] = (),
) -> str:
    """분석 설정의 결정적 해시.

    ``configuration_hash`` 는 실험 설정만 반영하므로, notebook 에서
    ``cut_slack_ratio`` 나 bit sweep 을 바꿔도 값이 그대로다. 그러면 merge key 가
    같아져 **서로 다른 분석 결과가 덮어써진다.** 이 해시를 merge key 에 함께 넣는다.
    """
    import hashlib
    import json as _json

    payload = {
        "analysis_schema_version": 2,
        "cut_slack_ratio": float(cut_slack_ratio),
        "reference_bits": int(reference_bits),
        "bit_sweep": sorted(int(b) for b in bit_sweep),
        "snapshot_policy": str(snapshot_policy),
        "snapshot_iterations": sorted(set(int(i) for i in snapshot_iterations)),
    }
    blob = _json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def select_analysis_results(
    frame: pd.DataFrame,
    analysis_hash: str,
    *,
    trajectory_hashes: dict[tuple[str, str], str] | None = None,
    expected_inputs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """요약과 전 구간 결과에서 동일 분석 설정 및 동일 입력만 선택한다.

    입력이 갱신되어 여러 실행이 공존하면 임의의 최신 행을 선택하지 않는다.
    ``trajectory_hashes[(source, instance_id)]``로 명시적으로 선택해야 한다.
    """
    identity = ["trajectory_source", "instance_id", "trajectory_hash"]
    required = [*identity, "analysis_hash", "configuration_hash"]
    if any(column not in frame for column in required):
        raise ValueError("분석 식별자 열이 누락되어 있다. analysis_v2 결과를 생성하라")
    selected = frame[frame["analysis_hash"] == analysis_hash].copy()
    for (source, instance_id), digest in (trajectory_hashes or {}).items():
        mask = (selected["trajectory_source"] == source) & (selected["instance_id"] == instance_id)
        selected = selected[~mask | (selected["trajectory_hash"] == digest)]
    if expected_inputs is not None:
        expected = expected_inputs[identity].drop_duplicates()
        selected = selected.merge(expected, on=identity, how="inner", validate="many_to_one")
        found = selected[identity].drop_duplicates()
        missing = expected.merge(found, on=identity, how="left", indicator=True)
        if (missing["_merge"] == "left_only").any():
            raise ValueError("최종 요약과 동일한 trajectory의 전 구간 결과가 누락되어 있다")
    if selected.empty or selected[required].isna().any().any():
        raise ValueError("선택한 분석 결과가 없거나 식별자 값이 누락되어 있다")
    if selected["configuration_hash"].nunique() != 1:
        raise ValueError("서로 다른 configuration_hash가 섞여 있다")
    mixed = selected.groupby(["trajectory_source", "instance_id"])["trajectory_hash"].nunique()
    if (mixed > 1).any():
        raise ValueError("입력 trajectory가 여러 개다. TRAJECTORY_HASHES로 하나를 선택하라")
    keys = [k for k in (
        *identity, "experiment", "bound_strategy", "theta_bits", "snapshot", "benders_iteration"
    ) if k in selected]
    if selected.duplicated(keys).any():
        raise ValueError("같은 분석 결과의 중복 행이 있다")
    return selected
