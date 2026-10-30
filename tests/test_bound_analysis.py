"""원본 실행 시점 복원과 분석 설정 분리를 검증한다."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cflp_bd_qa_ex2.data.io import load_instance
from cflp_bd_qa_ex2.encoding.bounds import build_bound
from cflp_bd_qa_ex2.evaluation.bound_analysis import (
    CutRecord, ObservedIteration, active_cuts_at, analysis_config_hash,
    analyze_trajectory, load_observed_trajectory, select_analysis_results,
    verify_snapshot_qubo,
)


def _write_iterations(tmp_path, rows):
    """실패 행과 cut 미추가 행을 포함한 QA CSV를 만든다."""
    target = tmp_path / "qa"
    target.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["instance_id"] = "example"
    frame["configuration_hash"] = "cfg"
    frame["graph_id"] = "graph"
    frame.to_csv(target / "qa_iterations_pegasus.csv", index=False)


def test_observed_iterations_preserve_failure_and_duplicate(tmp_path, small_instance, pilot_config):
    """마지막 dual이 없어도 실패 MP를 보존하고, 중복 cut iteration도 유지한다."""
    vector = str([0.0] * small_instance.n_facilities)
    _write_iterations(tmp_path, [
        {"benders_iteration": 1, "num_benders_cuts": 0, "raw_dual_u": "[1]", "raw_dual_v": vector},
        {"benders_iteration": 2, "num_benders_cuts": 1, "raw_dual_u": "[1]", "raw_dual_v": vector},
        {"benders_iteration": 3, "num_benders_cuts": 1, "raw_dual_u": "[2]", "raw_dual_v": vector},
        {"benders_iteration": 4, "num_benders_cuts": 2, "raw_dual_u": None, "raw_dual_v": None},
    ])
    trajectory = load_observed_trajectory(tmp_path, "pegasus", "example")
    assert trajectory.last_iteration == 4
    assert len(active_cuts_at(trajectory.cuts, 4)) == 2
    result = analyze_trajectory(
        small_instance, build_bound("S0", small_instance, pilot_config), trajectory.cuts,
        theta_bits=10, observations=trajectory.observations,
    )
    assert [r["benders_iteration"] for r in result] == [1, 2, 3, 4]
    assert [r["num_cuts"] for r in result] == [0, 1, 1, 2]
    assert result[1]["logical_variables"] == result[2]["logical_variables"]


def test_missing_activated_dual_is_rejected(tmp_path):
    """Cut 수가 늘었는데 dual을 복원할 수 없다면 중단한다."""
    _write_iterations(tmp_path, [
        {"benders_iteration": 1, "num_benders_cuts": 0, "raw_dual_u": None, "raw_dual_v": None},
        {"benders_iteration": 2, "num_benders_cuts": 1, "raw_dual_u": None, "raw_dual_v": None},
    ])
    with pytest.raises(ValueError, match="cut 복원 불일치"):
        load_observed_trajectory(tmp_path, "pegasus", "example")


@pytest.mark.parametrize("source", ["gurobi", "sa", "pegasus", "zephyr"])
@pytest.mark.parametrize("suffix", [81, 82, 83, 84, 91])
def test_benchmark_cut_counts_match_every_observed_mp(source, suffix, pilot_config):
    """저장된 20개 trajectory의 모든 실행 MP를 원본 cut 수와 대조한다."""
    root = Path(__file__).resolve().parents[1]
    tid = f"25x50_b{suffix}"
    required = (
        "benders/benders_iterations.csv" if source == "gurobi" else
        "sa/sa_iterations.csv" if source == "sa" else f"qa/qa_iterations_{source}.csv"
    )
    if not (root / "results" / required).exists() or not (root / f"data/benchmark/{tid}.json").exists():
        pytest.skip("원본 실험 데이터가 없는 설치 환경에서는 통합 검증을 생략한다")
    inst = load_instance(root / f"data/benchmark/{tid}.json")
    trajectory = load_observed_trajectory(root / "results", source, tid)
    rows = analyze_trajectory(
        inst, build_bound("S0", inst, pilot_config), trajectory.cuts,
        theta_bits=10, observations=trajectory.observations,
    )
    assert len(rows) == len(trajectory.observations)
    assert [(r["benders_iteration"], r["num_cuts"]) for r in rows] == [
        (o.iteration, o.num_cuts) for o in trajectory.observations
    ]
    if source == "gurobi":
        assert rows[-1]["benders_iteration"] == 1000
        assert rows[-1]["num_cuts"] == 999
    if source == "pegasus" and suffix == 81:
        assert rows[-1]["benders_iteration"] == 25
        assert rows[-1]["num_cuts"] == 24
        assert rows[-1]["logical_variables"] == 352


def test_snapshot_uses_requested_ratio_and_preserves_config(small_instance, pilot_config):
    """Ratio override가 실제 QUBO에도 적용되고 원래 cfg는 보존된다."""
    original = copy.deepcopy(pilot_config)
    cut = CutRecord(1, np.ones(small_instance.n_customers), np.zeros(small_instance.n_facilities), True)
    bound = build_bound("S0", small_instance, pilot_config)
    reports = [verify_snapshot_qubo(
        small_instance, bound, [cut], pilot_config, theta_bits=10, cut_slack_ratio=ratio,
    ) for ratio in (0.25, 0.5)]
    assert all(r["variables_match"] and r["edges_match"] for r in reports)
    assert reports[0]["actual_variables"] > reports[1]["actual_variables"]
    assert pilot_config == original


def test_analysis_hash_includes_actual_snapshot_list():
    """같은 정책명이어도 snapshot 목록이 다르면 다른 분석이다."""
    kwargs = dict(cut_slack_ratio=0.25, reference_bits=10, bit_sweep=[4, 6, 10], snapshot_policy="early")
    assert analysis_config_hash(**kwargs, snapshot_iterations=[1, 5]) != analysis_config_hash(
        **kwargs, snapshot_iterations=[1, 10])


def test_comparison_uses_same_inputs_for_summary_and_iterations():
    """전 구간에 옛 trajectory가 남아 있어도 선택한 요약과 같은 입력만 쓴다."""
    row = dict(trajectory_source="pegasus", instance_id="example", analysis_hash="a",
               configuration_hash="cfg", experiment="E1-A", bound_strategy="S0", theta_bits=10,
               snapshot="per_iteration", benders_iteration=1)
    frame = pd.DataFrame([{**row, "trajectory_hash": "old"}, {**row, "trajectory_hash": "new"}])
    with pytest.raises(ValueError, match="여러 개"):
        select_analysis_results(frame, "a")
    expected = pd.DataFrame([dict(trajectory_source="pegasus", instance_id="example", trajectory_hash="new")])
    selected = select_analysis_results(frame, "a", expected_inputs=expected)
    assert selected.trajectory_hash.tolist() == ["new"]
    with pytest.raises(ValueError, match="누락"):
        select_analysis_results(frame.iloc[:1], "a", expected_inputs=expected)
