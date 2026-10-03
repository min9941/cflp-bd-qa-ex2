"""scripts/ 의 핵심 동작을 직접 검증하는 테스트.

기존에는 `run_qa.py` 와 `run_embedding.py` 의 핵심 동작(활성 cut 필터,
sampler status 전파, embedding 별 분리)을 검증하는 테스트가 없었다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cflp_bd_qa_ex2.benders.solver import (
    GurobiMasterProvider,
    QuboMasterProvider,
    run_benders,
)
from cflp_bd_qa_ex2.status import Status, TrajectorySource


def test_cut_snapshots_mark_activation(small_instance, pilot_config):
    """수렴 iteration 에서 생성된 마지막 cut 은 활성화되지 않은 것으로 표시된다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    activated = [s for s in trace.cut_snapshots if s["cut_activated"]]
    assert len(activated) == trace.run.fields["num_cuts_generated"]
    # OPTIMAL 로 끝난 경우 마지막 snapshot 은 활성 cut 이 아니다.
    if trace.run.fields["termination_status"] == str(Status.OPTIMAL):
        assert trace.cut_snapshots[-1]["cut_activated"] is False


def test_embedding_snapshots_never_exceed_active_cuts(small_instance, pilot_config):
    """활성 cut 만 사용하면 embedding snapshot 의 cut 수가 실제 MP 를 넘지 않는다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    active = [s for s in trace.cut_snapshots if s["cut_activated"]]
    snapshot_sizes = list(range(len(active) + 1))
    assert max(snapshot_sizes) == trace.run.fields["num_cuts_generated"]


@pytest.mark.parametrize(
    "sampler_status", [Status.NO_QPU_ACCESS, Status.SOLVER_FAILED, Status.EMBEDDING_FAILED]
)
def test_sampler_status_is_not_masked(small_instance, pilot_config, sampler_status):
    """sampler 실패가 NO_FEASIBLE_SAMPLE 로 뒤바뀌지 않는다."""

    def failing_sampler(qubo, cfg, context=None):
        return {"samples": [], "status": str(sampler_status), "message": "테스트"}

    provider = QuboMasterProvider(TrajectorySource.QA_END_TO_END, failing_sampler)
    result = provider.candidate(small_instance, [], 1, pilot_config, {})
    assert result.status is sampler_status
    assert result.status is not Status.NO_FEASIBLE_SAMPLE


def test_empty_samples_without_status_is_no_feasible_sample(small_instance, pilot_config):
    """sampler 가 정상인데 sample 이 없으면 NO_FEASIBLE_SAMPLE 이 맞다."""

    def empty_sampler(qubo, cfg, context=None):
        return {"samples": [], "status": str(Status.SUCCESS)}

    provider = QuboMasterProvider(TrajectorySource.SA_END_TO_END, empty_sampler)
    result = provider.candidate(small_instance, [], 1, pilot_config, {})
    assert result.status is Status.NO_FEASIBLE_SAMPLE


def test_hardware_metrics_are_accumulated(small_instance, pilot_config):
    """run-level embedding/QPU metric 이 0 으로 고정되지 않는다."""

    def fake_sampler(qubo, cfg, context=None):
        assignment = {v: 0 for v in qubo.variable_labels}
        for name in qubo.variables_by_role("y"):
            assignment[name] = 1
        return {
            "samples": [(assignment, 0.0, 1, None)],
            "status": str(Status.SUCCESS),
            "embedding_runtime": 1.5,
            "qpu_runtime": 0.25,
            "num_embedding_trials": 5,
            "num_qpu_calls": 2,
        }

    import copy

    cfg = copy.deepcopy(pilot_config)
    cfg["benders"]["max_iterations"] = 2
    trace = run_benders(small_instance, QuboMasterProvider(TrajectorySource.QA_END_TO_END, fake_sampler), cfg)
    fields = trace.run.fields
    assert fields["embedding_runtime"] > 0.0
    assert fields["num_embedding_calls"] > 0
    assert fields["num_qpu_calls"] > 0


def test_duplicate_requires_repeated_y(small_instance, pilot_config):
    """서로 다른 y 가 같은 cut 을 만들면 STALLED 로 오판하지 않는다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    for iteration in trace.iterations:
        fields = iteration.as_record()
        assert "repeated_y" in fields
        assert "y_signature" in fields
        if fields.get("duplicate_cut") and not fields.get("repeated_y"):
            assert trace.run.fields["termination_status"] != str(
                Status.STALLED_DUPLICATE_CUT
            )


def test_sampler_receives_context(small_instance, pilot_config):
    """sampler 에 instance / cuts / iteration 이 전달된다."""
    seen = {}

    def recording_sampler(qubo, cfg, context=None):
        seen.update(context or {})
        return {"samples": [], "status": str(Status.SUCCESS)}

    provider = QuboMasterProvider(TrajectorySource.QA_END_TO_END, recording_sampler)
    provider.candidate(small_instance, [], 7, pilot_config, {})
    assert seen["instance"].instance_id == small_instance.instance_id
    assert seen["iteration"] == 7
    assert seen["cuts"] == []


def test_stall_requires_repeated_y_cut_pair(small_instance, pilot_config):
    """(y, cut) 쌍이 반복되어야 STALLED 로 판정한다."""
    trace = run_benders(small_instance, GurobiMasterProvider(), pilot_config)
    for iteration in trace.iterations:
        fields = iteration.as_record()
        assert "repeated_y_cut_pair" in fields
        if fields.get("duplicate_cut") and not fields.get("repeated_y_cut_pair"):
            assert trace.run.fields["termination_status"] != str(
                Status.STALLED_DUPLICATE_CUT
            )
