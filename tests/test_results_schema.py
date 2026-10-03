"""결과 schema 와 provenance 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from cflp_bd_qa_ex2.benders.solver import GurobiMasterProvider, run_benders
from cflp_bd_qa_ex2.benders.trace import RECORD_TYPE_ITERATION, RECORD_TYPE_RUN
from cflp_bd_qa_ex2.evaluation.results import attach_provenance, provenance, save_records
from cflp_bd_qa_ex2.status import DualGuardAction, DualMinimalityStatus, Status

REQUIRED_RUN_FIELDS = (
    "benders_iterations",
    "benders_runtime",
    "mp_runtime",
    "continuous_mp_certificate_runtime",
    "encoded_mp_runtime",
    "qubo_runtime",
    "sp_runtime",
    "embedding_runtime",
    "qpu_runtime",
    "num_mp_calls",
    "num_sp_calls",
    "num_embedding_calls",
    "num_qpu_calls",
    "final_objective",
    "final_optimality_gap",
    "certified_lower_bound",
    "best_upper_bound",
    "termination_status",
)

REQUIRED_PROVENANCE_FIELDS = (
    "project_version",
    "git_commit",
    "python_version",
    "package_versions",
    "configuration_hash",
)


@pytest.fixture(scope="module")
def trace(small_instance, pilot_config):
    """Gurobi-controlled Benders trace."""
    return run_benders(small_instance, GurobiMasterProvider(), pilot_config)


def test_run_record_has_required_fields(trace):
    """Benders-level 기록에 필수 field 가 모두 있다."""
    record = trace.run.as_record()
    missing = [f for f in REQUIRED_RUN_FIELDS if f not in record]
    assert missing == []


def test_status_fields_are_separate(trace):
    """dual guard 진단이 termination_status 를 덮어쓰지 않는다."""
    record = trace.run.as_record()
    assert record["termination_status"] == str(Status.OPTIMAL)
    assert record["dual_minimality_status"] == str(DualMinimalityStatus.MINIMAL)
    assert record["dual_guard_action"] == str(DualGuardAction.NONE)
    assert record["dual_minimality_status"] != record["termination_status"]


def test_record_types_distinguish_levels(trace):
    """iteration 과 run 이 다른 record_type 을 갖는다."""
    records = trace.to_records()
    assert records[0]["record_type"] == RECORD_TYPE_ITERATION
    assert records[-1]["record_type"] == RECORD_TYPE_RUN


def test_iteration_record_separates_objectives(trace):
    """QUBO energy / MP objective / SP value / original objective 를 분리 저장한다."""
    record = trace.iterations[0].as_record()
    for field in (
        "continuous_mp_objective",
        "true_sp_value_for_decoded_y",
        "candidate_original_objective",
        "certified_lower_bound",
        "best_upper_bound",
    ):
        assert field in record
    assert "decoded_x" not in record  # MP 에는 x 가 없다


def test_provenance_fields(pilot_config):
    """provenance 에 필수 field 가 있다."""
    record = provenance(pilot_config)
    for field in REQUIRED_PROVENANCE_FIELDS:
        assert field in record


def test_attach_provenance(trace, pilot_config):
    """모든 레코드에 provenance 가 붙는다."""
    records = attach_provenance(trace.to_records(), pilot_config)
    assert all("configuration_hash" in r for r in records)


def test_save_records_csv_and_merge(tmp_path, pilot_config):
    """CSV 저장과 key 기준 병합이 동작한다."""
    first = [{"instance_id": "a", "value": 1}, {"instance_id": "b", "value": 2}]
    save_records(first, tmp_path, "demo", formats=("csv",), merge_keys=("instance_id",))

    second = [{"instance_id": "a", "value": 99}]
    save_records(second, tmp_path, "demo", formats=("csv",), merge_keys=("instance_id",))

    frame = pd.read_csv(tmp_path / "demo.csv")
    assert len(frame) == 2
    assert int(frame.loc[frame["instance_id"] == "a", "value"].iloc[0]) == 99


def test_failures_are_not_deleted(small_instance, pilot_config):
    """실패 상태를 성공으로 바꾸거나 행을 지우지 않는다."""
    import copy

    cfg = copy.deepcopy(pilot_config)
    cfg["benders"]["max_iterations"] = 1
    trace = run_benders(small_instance, GurobiMasterProvider(), cfg)
    record = trace.run.as_record()
    assert record["termination_status"] == str(Status.MAX_ITERATIONS)
    assert len(trace.iterations) == 1
