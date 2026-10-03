"""리뷰에서 추가된 동작을 직접 검증하는 테스트.

preflight 영속화, ``qa_requested``/``used_for_qa`` 분리, figure 덮어쓰기 방지,
``git_dirty`` provenance, heatmap micro/macro 구분, QA sampler 의 실패 보존을
수동 확인에만 의존하지 않도록 한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cflp_bd_qa_ex2.evaluation.plots import (
    embedding_success_table,
    plot_bound_trajectory,
    plot_embedding_success_heatmap,
)
from cflp_bd_qa_ex2.evaluation.results import git_state, provenance
from cflp_bd_qa_ex2.status import Status


def test_provenance_reports_git_dirty(pilot_config):
    """provenance 가 working tree 의 청결 여부를 숨기지 않는다."""
    record = provenance(pilot_config)
    assert "git_dirty" in record
    assert "git_diff_hash" in record
    state = git_state()
    if state["git_commit"] is not None:
        assert state["git_dirty"] in (True, False)
        if state["git_dirty"]:
            assert state["git_diff_hash"] is not None


def test_bound_trajectory_suffix_creates_distinct_files(tmp_path):
    """suffix=True 면 인스턴스별로 다른 파일이 생성된다."""
    frame = pd.DataFrame(
        {
            "instance_id": ["a", "a", "b", "b"],
            "trajectory_source": ["gurobi_controlled"] * 4,
            "benders_iteration": [1, 2, 1, 2],
            "certified_lower_bound": [1.0, 2.0, 1.0, 2.0],
            "best_upper_bound": [3.0, 3.0, 3.0, 3.0],
        }
    )
    first = plot_bound_trajectory(frame, tmp_path, "a", suffix=True)
    second = plot_bound_trajectory(frame, tmp_path, "b", suffix=True)
    assert first != second
    assert first.exists() and second.exists()

    fixed = plot_bound_trajectory(frame, tmp_path, "a", suffix=False)
    assert fixed.name == "fig03_bound_trajectory.png"


def test_heatmap_micro_and_macro_are_distinct(tmp_path):
    """micro 와 macro 평균이 다른 파일로 저장되고 값이 달라질 수 있다."""
    # instance s1 은 iteration 이 많고 전부 실패, s2 는 1개이고 성공.
    rows = [
        {"instance_name": "4x12", "instance_id": "s1", "topology_label": "P16",
         "benders_iteration": i, "success": False}
        for i in range(4)
    ] + [
        {"instance_name": "4x12", "instance_id": "s2", "topology_label": "P16",
         "benders_iteration": 0, "success": True}
    ]
    frame = pd.DataFrame(rows)

    micro = plot_embedding_success_heatmap(frame, tmp_path, average="micro")
    macro = plot_embedding_success_heatmap(frame, tmp_path, average="macro")
    assert micro != macro
    assert micro.exists() and macro.exists()

    # plot 이 실제로 사용하는 집계 함수를 직접 검증한다.
    micro_table = embedding_success_table(frame, average="micro")
    macro_table = embedding_success_table(frame, average="macro")
    assert micro_table.loc["4x12", "P16"] == pytest.approx(0.2)
    assert macro_table.loc["4x12", "P16"] == pytest.approx(0.5)

    with pytest.raises(ValueError):
        embedding_success_table(frame, average="median")


def test_git_diff_hash_detects_untracked_content(tmp_path, monkeypatch):
    """untracked 파일의 **내용**이 바뀌면 git_diff_hash 도 바뀐다."""
    import subprocess

    from cflp_bd_qa_ex2.evaluation import results as results_module

    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *args: subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    run("init")
    run("config", "user.email", "a@b")
    run("config", "user.name", "c")
    (repo / "tracked.txt").write_text("base", encoding="utf-8")
    run("add", "-A")
    run("commit", "-m", "init")

    clean = results_module.git_state(repo)
    if clean["git_commit"] is None:
        pytest.skip("git 을 사용할 수 없다")
    assert clean["git_dirty"] is False

    extra = repo / "untracked.txt"
    extra.write_text("first", encoding="utf-8")
    first = results_module.git_state(repo)
    assert first["git_dirty"] is True

    extra.write_text("second", encoding="utf-8")
    second = results_module.git_state(repo)
    assert second["git_dirty"] is True
    # 파일명은 같고 내용만 달라졌는데도 해시가 달라져야 한다.
    assert first["git_diff_hash"] != second["git_diff_hash"]

    # 내용을 되돌리면 해시도 되돌아온다.
    extra.write_text("first", encoding="utf-8")
    assert results_module.git_state(repo)["git_diff_hash"] == first["git_diff_hash"]


def test_git_diff_hash_handles_non_ascii_paths(tmp_path):
    """한글 등 비ASCII 파일명의 **내용** 변경도 감지한다.

    Git 은 기본적으로 비ASCII 경로를 이스케이프해 출력하므로,
    그대로 읽으면 실제 파일을 찾지 못해 내용 변경을 놓친다.
    """
    import subprocess

    from cflp_bd_qa_ex2.evaluation import results as results_module

    repo = tmp_path / "repo_ko"
    repo.mkdir()
    run = lambda *args: subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    run("init")
    run("config", "user.email", "a@b")
    run("config", "user.name", "c")
    (repo / "tracked.txt").write_text("base", encoding="utf-8")
    run("add", "-A")
    run("commit", "-m", "init")

    if results_module.git_state(repo)["git_commit"] is None:
        pytest.skip("git 을 사용할 수 없다")

    korean = repo / "한글.txt"
    korean.write_text("first", encoding="utf-8")
    first = results_module.git_state(repo)
    assert first["git_dirty"] is True
    assert first["git_diff_hash"] is not None

    korean.write_text("second", encoding="utf-8")
    second = results_module.git_state(repo)
    assert first["git_diff_hash"] != second["git_diff_hash"]

    korean.write_text("first", encoding="utf-8")
    assert results_module.git_state(repo)["git_diff_hash"] == first["git_diff_hash"]


def test_qa_master_solver_preserves_failed_embeddings(small_instance, pilot_config, monkeypatch):
    """embedding 이 전부 실패해도 trial 행을 보존하고 상태를 정확히 돌려준다."""
    from cflp_bd_qa_ex2.embedding.embedder import EmbeddingTrial
    from cflp_bd_qa_ex2.embedding.targets import build_ideal_target
    from cflp_bd_qa_ex2.qubo.builder import build_master_qubo
    from cflp_bd_qa_ex2.solvers import qa_master

    def all_failed(source, target, cfg):
        return [
            EmbeddingTrial(
                success=False,
                embedding_status=Status.EMBEDDING_FAILED,
                embedding_seed=seed,
                embedding_time=0.1,
                message="주어진 search budget 에서 embedding 을 찾지 못함",
                fields={},
            )
            for seed in cfg["embedding"]["seeds"]
        ]

    monkeypatch.setattr(qa_master, "run_embedding_trials", all_failed)

    target = build_ideal_target("pegasus", 16, "test_target")
    qa_records: list[dict] = []
    embedding_records: list[dict] = []
    solver = qa_master.QAMasterProblemSolver(
        target=target,
        instance_id=small_instance.instance_id,
        cfg=pilot_config,
        qa_records=qa_records,
        embedding_records=embedding_records,
    )

    qubo = build_master_qubo(small_instance, [], pilot_config)
    result = solver(
        qubo, pilot_config,
        {"instance": small_instance, "cuts": [], "iteration": 1},
    )

    assert result["status"] == str(Status.EMBEDDING_FAILED)
    assert result["samples"] == []
    # 실패한 seed 가 전부 보존된다.
    assert len(embedding_records) == len(pilot_config["embedding"]["seeds"])
    for record in embedding_records:
        assert record["qa_requested"] is True
        assert record["used_for_qa"] is False  # 실패는 QA 에 쓰이지 않는다
        assert record["embedding_file"] is None
        assert record["source_graph_fingerprint"]
    assert qa_records == []  # QPU 제출이 없었으므로 평가 행도 없다


def test_git_state_survives_non_utf8_bytes(tmp_path):
    """git 출력에 로캘로 디코딩할 수 없는 바이트가 있어도 실패하지 않는다.

    Windows 한국어 환경의 기본 인코딩은 cp949 이므로, UTF-8 로 기록된
    한국어 주석이 든 diff 를 ``text=True`` 로 받으면 UnicodeDecodeError 가 난다.
    """
    import subprocess

    from cflp_bd_qa_ex2.evaluation import results as results_module

    repo = tmp_path / "repo_bytes"
    repo.mkdir()
    run = lambda *args: subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    run("init")
    run("config", "user.email", "a@b")
    run("config", "user.name", "c")
    source = repo / "source.py"
    source.write_text("# 한국어 주석\nVALUE = 1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-m", "init")

    if results_module.git_state(repo)["git_commit"] is None:
        pytest.skip("git 을 사용할 수 없다")

    # tracked diff 에 한국어가 들어가고, untracked 에는 디코딩 불가 바이트를 둔다.
    source.write_text("# 한국어 주석 변경\nVALUE = 2\n", encoding="utf-8")
    (repo / "binary.dat").write_bytes(b"\x9c\xff\xfe raw bytes")

    state = results_module.git_state(repo)
    assert state["git_dirty"] is True
    assert isinstance(state["git_diff_hash"], str)

    # 내용이 바뀌면 해시도 바뀐다.
    source.write_text("# 한국어 주석 재변경\nVALUE = 3\n", encoding="utf-8")
    assert results_module.git_state(repo)["git_diff_hash"] != state["git_diff_hash"]


def test_decode_handles_invalid_bytes():
    """``_decode`` 가 잘못된 바이트에서 예외를 던지지 않는다."""
    from cflp_bd_qa_ex2.evaluation.results import _decode

    assert _decode(None) == ""
    assert _decode(b"") == ""
    assert _decode("한국어".encode("utf-8")) == "한국어"
    assert isinstance(_decode(b"\x9c\xff"), str)  # cp949/utf-8 모두 불가한 바이트


def test_flatten_qpu_timing_columns_are_stable():
    """timing 이 없어도 열 구성이 흔들리지 않는다."""
    from cflp_bd_qa_ex2.solvers.qa import QPU_TIMING_KEYS, flatten_qpu_timing

    empty = flatten_qpu_timing(None)
    assert set(empty) == {f"qa_{key}_us" for key in QPU_TIMING_KEYS}
    assert all(value is None for value in empty.values())
    assert flatten_qpu_timing({}) == empty


def test_flatten_qpu_timing_keeps_microseconds():
    """값은 마이크로초 원본 그대로 유지한다."""
    from cflp_bd_qa_ex2.solvers.qa import flatten_qpu_timing

    flat = flatten_qpu_timing(
        {
            "qpu_access_time": 145678.0,
            "qpu_sampling_time": 123456.0,
            "qpu_anneal_time_per_sample": 20.0,
            "qpu_programming_time": 14500.0,
        }
    )
    assert flat["qa_qpu_access_time_us"] == pytest.approx(145678.0)
    assert flat["qa_qpu_sampling_time_us"] == pytest.approx(123456.0)
    assert flat["qa_qpu_anneal_time_per_sample_us"] == pytest.approx(20.0)
    # 보고되지 않은 표준 key 는 0 이 아니라 None 이다.
    assert flat["qa_qpu_readout_time_per_sample_us"] is None


def test_flatten_qpu_timing_handles_unknown_and_non_numeric():
    """표준 목록에 없는 숫자 key 는 추가하고, 비숫자는 제외한다."""
    from cflp_bd_qa_ex2.solvers.qa import flatten_qpu_timing

    flat = flatten_qpu_timing(
        {"vendor_specific_time": 7.5, "label": "text", "flag": True}
    )
    assert flat["qa_vendor_specific_time_us"] == pytest.approx(7.5)
    assert "qa_label_us" not in flat
    assert "qa_flag_us" not in flat  # bool 은 숫자로 취급하지 않는다


def test_qa_timing_dict_is_not_stored_raw():
    """timing 을 dict 통째로 저장하지 않는다 (CSV 에서 문자열이 되므로)."""
    from cflp_bd_qa_ex2.solvers.qa import flatten_qpu_timing

    assert "qa_timing" not in flatten_qpu_timing({"qpu_access_time": 1.0})


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Problem not accepted because user has insufficient remaining solver "
         "access time in project yXwP", "QPU_QUOTA_EXHAUSTED"),
        ("monthly quota exceeded", "QPU_QUOTA_EXHAUSTED"),
        ("401 Unauthorized", "NO_QPU_ACCESS"),
        ("chain strength invalid", "EMBEDDING_INVALID"),
        ("something else went wrong", "SOLVER_FAILED"),
    ],
)
def test_classify_qpu_exception(message, expected):
    """QPU 예외가 종류별로 다른 status 로 분류된다."""
    from cflp_bd_qa_ex2.solvers.qa import classify_qpu_exception

    status, text = classify_qpu_exception(RuntimeError(message))
    assert str(status) == expected
    assert message in text


def test_quota_error_is_not_no_qpu_access():
    """할당량 소진을 접근 불가로 기록하지 않는다."""
    from cflp_bd_qa_ex2.solvers.qa import classify_qpu_exception
    from cflp_bd_qa_ex2.status import Status

    status, _ = classify_qpu_exception(
        RuntimeError("user has insufficient remaining solver access time")
    )
    assert status is Status.QPU_QUOTA_EXHAUSTED
    assert status is not Status.NO_QPU_ACCESS
    assert status is not Status.SOLVER_FAILED


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Problem not accepted because user has insufficient remaining "
         "solver access time in project yXwP", "QPU_QUOTA_EXHAUSTED"),
        ("You have exceeded your monthly quota", "QPU_QUOTA_EXHAUSTED"),
        ("SolverAuthenticationError: invalid token", "NO_QPU_ACCESS"),
        ("Connection timeout while polling", "NO_QPU_ACCESS"),
        ("chain strength must be positive", "EMBEDDING_INVALID"),
        ("unexpected internal failure", "SOLVER_FAILED"),
    ],
)
def test_qpu_exception_classification(message, expected, monkeypatch, small_instance, pilot_config):
    """QPU 예외를 종류별로 구분한다 (전부 NO_QPU_ACCESS 로 뭉뚱그리지 않는다)."""
    from cflp_bd_qa_ex2.qubo.builder import build_master_qubo
    from cflp_bd_qa_ex2.solvers import qa as qa_module

    qubo = build_master_qubo(small_instance, [], pilot_config)

    class _FailingSampler:
        def __init__(self, *args, **kwargs):
            pass

        def sample(self, *args, **kwargs):
            raise RuntimeError(message)

    fake_system = type("m", (), {
        "DWaveSampler": _FailingSampler,
        "FixedEmbeddingComposite": lambda base, embedding: _FailingSampler(),
    })
    monkeypatch.setitem(
        __import__("sys").modules, "dwave.system", fake_system
    )
    monkeypatch.setattr(
        qa_module, "compute_chain_strength",
        lambda qubo, embedding, cfg: {"chain_strength": 1.0},
    )

    result = qa_module.run_qa(
        qubo, {v: [0] for v in qubo.variable_labels}, "solver", "graph",
        pilot_config, embedding_seed=2024,
    )
    assert result["status"] == expected
    assert result["samples"] == []
    # 예외를 밖으로 던지지 않는다.
    assert "message" in result


def test_qpu_lookup_accepts_profile_without_nameerror():
    """profile 인자를 받아야 하고, QPU 가 없어도 예외를 던지지 않는다.

    이전에 ``Client.from_config(profile=profile)`` 만 넣고 함수 시그니처에
    파라미터를 추가하지 않아 ``NameError`` 가 status 메시지로 새어 나갔다.
    """
    import inspect

    from cflp_bd_qa_ex2.embedding.targets import list_accessible_qpus

    assert "profile" in inspect.signature(list_accessible_qpus).parameters

    for argument in (None, "does_not_exist"):
        status, solvers, message = list_accessible_qpus(argument)
        assert isinstance(message, str)
        assert "not defined" in message or "not found" in message or solvers
        # 구현 오류가 status 메시지로 위장되면 안 된다.
        assert "is not defined" not in message


def test_pick_instances_and_size_seed_intersection(pilot_config):
    """sizes 와 seeds 가 AND 로 적용되고, 교집합이 비면 경고한다."""
    from cflp_bd_qa_ex2.data.generator import generate_all
    from cflp_bd_qa_ex2.evaluation.target_selection import describe_selection, pick_instances

    instances = generate_all(pilot_config)

    assert len(pick_instances(instances)) == 20
    assert pick_instances(instances, None, [100, 200, 300, 400]) == [
        "4x12_s100", "8x25_s200", "16x50_s300", "25x50_s400"
    ]
    # 크기 오름차순 정렬 (작은 것부터 실행해야 중단 시 결과가 남는다)
    sizes = [
        next(i for i in instances if i.instance_id == tid).n_facilities
        * next(i for i in instances if i.instance_id == tid).n_customers
        for tid in pick_instances(instances)
    ]
    assert sizes == sorted(sizes)

    # sizes 와 seeds 의 교집합이 비는 조합을 경고로 잡는다.
    info = describe_selection(instances, ["4x12", "8x25"], [100], "controlled")
    assert info["instance_ids"] == ["4x12_s100"]
    assert any("8x25" in w for w in info["warnings"])

    empty = describe_selection(instances, ["99x99"], None, "e2e")
    assert empty["instance_ids"] == []
    assert empty["warnings"]


def test_topology_target_mismatch_is_detected(tmp_path, pilot_config):
    """다른 topology 와 대상 인스턴스가 다르면 감지한다."""
    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.target_selection import compare_with_other_topology

    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    cfg_hash = pilot_config["configuration_hash"]

    # 비교 대상이 없으면 no_reference
    assert compare_with_other_topology(qa_dir, "zephyr", ["4x12_s100"], cfg_hash)["status"] == "no_reference"

    pd.DataFrame({
        "instance_id": ["4x12_s100", "8x25_s200"],
        "configuration_hash": [cfg_hash] * 2,
    }).to_csv(qa_dir / "qa_runs_pegasus.csv", index=False)

    same = compare_with_other_topology(qa_dir, "zephyr", ["4x12_s100", "8x25_s200"], cfg_hash)
    assert same["status"] == "match"

    diff = compare_with_other_topology(qa_dir, "zephyr", ["4x12_s100"], cfg_hash)
    assert diff["status"] == "mismatch"
    assert diff["only_there"] == ["8x25_s200"]
    assert diff["other_topology"] == "pegasus"


def test_qa_checkpoint_is_incremental_and_idempotent(tmp_path, pilot_config):
    """인스턴스마다 저장해도 행이 중복되지 않고, 중간 결과가 남는다."""
    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import save_qa_checkpoint

    def run_row(instance_id):
        return {
            "instance_id": instance_id, "trajectory_source": "qa_end_to_end",
            "graph_id": "g1", "termination_status": "OPTIMAL",
        }

    rows = [run_row("4x12_s100")]
    save_qa_checkpoint(tmp_path, "pegasus", pilot_config, [], [], [], rows)
    frame = pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv")
    assert len(frame) == 1  # 첫 인스턴스가 끝난 시점에 이미 디스크에 있다

    # 두 번째 인스턴스 후 다시 저장 (리스트는 누적된다)
    rows.append(run_row("8x25_s200"))
    save_qa_checkpoint(tmp_path, "pegasus", pilot_config, [], [], [], rows)
    frame = pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv")
    assert len(frame) == 2  # 중복되지 않는다

    # 같은 내용을 또 저장해도 늘지 않는다 (마지막 저장 셀 재실행 대비)
    save_qa_checkpoint(tmp_path, "pegasus", pilot_config, [], [], [], rows)
    assert len(pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv")) == 2

    # topology 가 다르면 다른 파일에 저장된다
    save_qa_checkpoint(tmp_path, "zephyr", pilot_config, [], [], [], [run_row("4x12_s100")])
    assert (tmp_path / "qa" / "qa_runs_zephyr.csv").exists()
    assert len(pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv")) == 2


def test_git_state_reports_unknown_when_command_fails(tmp_path):
    """git 명령이 실패하면 깨끗한 tree 로 잘못 기록하지 않는다."""
    from cflp_bd_qa_ex2.evaluation import results as results_module

    # git 저장소가 아닌 경로 -> commit 조회부터 실패
    state = results_module.git_state(tmp_path)
    assert state["git_commit"] is None
    assert state["git_dirty"] is None
    assert state["git_diff_hash"] is None
    # False 로 잘못 기록되면 안 된다
    assert state["git_dirty"] is not False


def test_qa_checkpoint_preserves_provenance_in_production_shape(tmp_path, pilot_config):
    """production 과 **같은 방식**으로 호출했을 때 provenance 가 보존된다.

    notebook 과 CLI 는 raw record 를 누적하고 checkpoint 함수만 부른다.
    테스트가 미리 ``attach_provenance_once()`` 를 부르면 실제 경로를 검증하지 못한다.
    """
    import time

    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import save_qa_checkpoint

    def run_row(instance_id):
        return {
            "instance_id": instance_id, "trajectory_source": "qa_end_to_end",
            "graph_id": "g1", "termination_status": "OPTIMAL",
        }

    # production 과 동일: raw record 누적 -> checkpoint 호출만 한다
    rows = [run_row("4x12_s100")]
    save_qa_checkpoint(tmp_path, "pegasus", pilot_config, [], [], [], rows)
    first = pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv").set_index("instance_id")
    first_timestamp = first.loc["4x12_s100", "timestamp"]
    assert len(first) == 1  # 첫 인스턴스가 끝난 시점에 이미 디스크에 있다

    time.sleep(0.05)
    rows.append(run_row("8x25_s200"))
    save_qa_checkpoint(tmp_path, "pegasus", pilot_config, [], [], [], rows)

    frame = pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv").set_index("instance_id")
    assert len(frame) == 2
    assert frame.loc["4x12_s100", "timestamp"] == first_timestamp  # 덮어쓰이지 않는다
    assert frame.loc["8x25_s200", "timestamp"] != first_timestamp

    # 같은 내용을 또 저장해도 늘지 않고 timestamp 도 그대로다
    save_qa_checkpoint(tmp_path, "pegasus", pilot_config, [], [], [], rows)
    again = pd.read_csv(tmp_path / "qa" / "qa_runs_pegasus.csv").set_index("instance_id")
    assert len(again) == 2
    assert again.loc["4x12_s100", "timestamp"] == first_timestamp


def test_embedding_checkpoint_preserves_provenance_in_production_shape(tmp_path, pilot_config):
    """controlled embedding 도 같은 방식으로 provenance 가 보존된다."""
    import time

    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import save_embedding_checkpoint

    def row(instance_id):
        return {
            "instance_id": instance_id, "trajectory_source": "gurobi_controlled",
            "benders_iteration": 0, "topology_label": "P16", "graph_id": "g",
            "embedding_seed": 2024, "success": True,
        }

    records = [row("4x12_s100")]
    save_embedding_checkpoint(tmp_path, pilot_config, records)
    path = tmp_path / "embedding" / "embedding_trials.csv"
    first_timestamp = pd.read_csv(path).set_index("instance_id").loc["4x12_s100", "timestamp"]

    time.sleep(0.05)
    records.append(row("8x25_s200"))
    save_embedding_checkpoint(tmp_path, pilot_config, records)

    frame = pd.read_csv(path).set_index("instance_id")
    assert len(frame) == 2
    assert frame.loc["4x12_s100", "timestamp"] == first_timestamp
    assert frame.loc["8x25_s200", "timestamp"] != first_timestamp


def test_attach_provenance_once_updates_in_place(pilot_config):
    """원본 dict 를 제자리에서 갱신해야 다음 호출에서 보존된다."""
    from cflp_bd_qa_ex2.evaluation.results import attach_provenance_once

    rows = [{"id": 1}]
    attach_provenance_once(rows, pilot_config)
    assert "timestamp" in rows[0]  # 반환값이 아니라 **원본**이 갱신된다

    stamp = rows[0]["timestamp"]
    rows.append({"id": 2})
    attach_provenance_once(rows, pilot_config)
    assert rows[0]["timestamp"] == stamp
    assert rows[1]["timestamp"] != stamp


def test_load_qa_results_ignores_legacy_when_suffixed_exists(tmp_path):
    """topology 파일이 있으면 구형 무접미사 파일을 무시한다."""
    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import load_qa_results

    pd.DataFrame({"instance_id": ["a"], "v": [1]}).to_csv(tmp_path / "qa_runs.csv", index=False)
    # 구형 파일만 있으면 그것을 읽는다
    assert len(load_qa_results(tmp_path, "qa_runs")) == 1

    pd.DataFrame({"instance_id": ["a"], "v": [1]}).to_csv(
        tmp_path / "qa_runs_pegasus.csv", index=False)
    pd.DataFrame({"instance_id": ["a"], "v": [1]}).to_csv(
        tmp_path / "qa_runs_zephyr.csv", index=False)

    frame = load_qa_results(tmp_path, "qa_runs")
    assert len(frame) == 2  # 구형 1행이 더해져 3행이 되면 안 된다
    assert load_qa_results(tmp_path, "does_not_exist").empty


def test_embedding_checkpoint_is_incremental(tmp_path, pilot_config):
    """controlled embedding trial 도 증분 저장되고 중복되지 않는다."""
    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import save_embedding_checkpoint

    def row(instance_id, seed):
        return {
            "instance_id": instance_id, "trajectory_source": "gurobi_controlled",
            "benders_iteration": 0, "topology_label": "P16", "graph_id": "g",
            "embedding_seed": seed, "success": True,
        }

    records = [row("4x12_s100", 2024)]
    assert save_embedding_checkpoint(tmp_path, pilot_config, records) == 1
    assert len(pd.read_csv(tmp_path / "embedding" / "embedding_trials.csv")) == 1

    records.append(row("8x25_s200", 2024))
    save_embedding_checkpoint(tmp_path, pilot_config, records)
    assert len(pd.read_csv(tmp_path / "embedding" / "embedding_trials.csv")) == 2

    save_embedding_checkpoint(tmp_path, pilot_config, records)  # 재저장
    assert len(pd.read_csv(tmp_path / "embedding" / "embedding_trials.csv")) == 2


def _diagnose_module():
    """scripts/diagnose_cuts.py 를 import 한다."""
    import sys
    from pathlib import Path as _Path

    scripts = _Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import diagnose_cuts

    return diagnose_cuts


def test_diagnose_separates_topologies():
    """같은 instance 라도 Pegasus 와 Zephyr 는 진단 행이 나뉜다."""
    import pandas as pd

    module = _diagnose_module()

    frame = pd.DataFrame([
        {"instance_id": "4x12_s100", "trajectory_source": "qa_end_to_end",
         "graph_id": "gP", "topology_label": "P16", "configuration_hash": "h1",
         "benders_iteration": 1, "cut_signature": "s1", "decoded_y": "[1,0]"},
        {"instance_id": "4x12_s100", "trajectory_source": "qa_end_to_end",
         "graph_id": "gP", "topology_label": "P16", "configuration_hash": "h1",
         "benders_iteration": 2, "cut_signature": "s1", "decoded_y": "[1,0]"},
        {"instance_id": "4x12_s100", "trajectory_source": "qa_end_to_end",
         "graph_id": "gZ", "topology_label": "Z12", "configuration_hash": "h1",
         "benders_iteration": 1, "cut_signature": "s2", "decoded_y": "[0,1]"},
    ])
    rows = module.diagnose(frame)

    assert len(rows) == 2  # 하나로 합쳐지면 안 된다
    by_topology = {r["topology_label"]: r for r in rows}
    assert by_topology["P16"]["num_cuts_generated"] == 2
    assert by_topology["P16"]["duplicate_pattern"] == "repeated_y"
    assert by_topology["Z12"]["num_cuts_generated"] == 1
    assert by_topology["Z12"]["duplicate_pattern"] == "none"


def test_diagnose_separates_configurations():
    """설정이 다르면 같은 instance 라도 진단 행이 나뉜다."""
    import pandas as pd

    module = _diagnose_module()

    frame = pd.DataFrame([
        {"instance_id": "4x12_s100", "trajectory_source": "sa_end_to_end",
         "configuration_hash": "h1", "benders_iteration": 1,
         "cut_signature": "s1", "decoded_y": "[1,0]"},
        {"instance_id": "4x12_s100", "trajectory_source": "sa_end_to_end",
         "configuration_hash": "h2", "benders_iteration": 1,
         "cut_signature": "s1", "decoded_y": "[1,0]"},
    ])
    assert len(module.diagnose(frame)) == 2


def test_diagnose_handles_missing_group_columns():
    """graph_id 가 없는 Gurobi/SA 결과도 정상 처리된다."""
    import pandas as pd

    module = _diagnose_module()

    frame = pd.DataFrame([
        {"instance_id": "4x12_s100", "trajectory_source": "gurobi_controlled",
         "benders_iteration": 1, "cut_signature": "s1", "continuous_mp_y": "[1,0]"},
        {"instance_id": "4x12_s100", "trajectory_source": "gurobi_controlled",
         "benders_iteration": 2, "cut_signature": "s1", "continuous_mp_y": "[1,0]"},
    ])
    rows = module.diagnose(frame)
    assert len(rows) == 1
    assert rows[0]["graph_id"] == module.MISSING
    assert rows[0]["duplicate_pattern"] == "repeated_y"
    # merge key 와 진단 record 의 구분 field 가 일치한다
    assert set(module.GROUP_FIELDS) <= set(rows[0])


def test_diagnose_rows_do_not_duplicate_across_saves(tmp_path, pilot_config):
    """저장-재로딩 왕복 후에도 같은 진단 행이 중복되지 않는다.

    빈 문자열로 채운 merge key 는 ``pd.read_csv`` 에서 ``NaN`` 이 되어
    다음 저장 때 key 가 달라진다. in-memory 그룹 분리만 검사하면 이를 놓친다.
    """
    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import attach_provenance, save_records

    module = _diagnose_module()

    frame = pd.DataFrame([
        {"instance_id": "4x12_s100", "trajectory_source": "gurobi_controlled",
         "benders_iteration": 1, "cut_signature": "s1", "continuous_mp_y": "[1,0]"},
    ])
    rows = module.diagnose(frame)
    assert rows[0]["graph_id"] == module.MISSING

    for _ in range(3):
        save_records(
            attach_provenance(rows, pilot_config), tmp_path, "diag",
            ("csv",), merge_keys=module.GROUP_FIELDS,
        )

    saved = pd.read_csv(tmp_path / "diag.csv")
    assert len(saved) == 1
    # 왕복해도 결측으로 바뀌지 않는다
    assert saved["graph_id"].notna().all()
    assert saved["graph_id"].iloc[0] == module.MISSING


def test_merge_key_missing_sentinel_is_not_read_as_na():
    """sentinel 이 pandas 결측으로 해석되지 않는다."""
    import io

    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import MERGE_KEY_MISSING

    # "NA", "N/A", "null" 등은 pandas 가 결측으로 읽으므로 쓰면 안 된다.
    frame = pd.read_csv(io.StringIO(f"key\n{MERGE_KEY_MISSING}\n"))
    assert frame["key"].notna().all()
    assert frame["key"].iloc[0] == MERGE_KEY_MISSING


def test_save_records_normalizes_missing_merge_keys(tmp_path):
    """구형 파일의 빈 칸과 새 sentinel 이 같은 행으로 병합된다."""
    import pandas as pd

    from cflp_bd_qa_ex2.evaluation.results import MERGE_KEY_MISSING, save_records

    keys = ("instance_id", "graph_id")
    pd.DataFrame([{"instance_id": "a", "graph_id": "", "value": 1}]).to_csv(
        tmp_path / "x.csv", index=False)

    save_records(
        [{"instance_id": "a", "graph_id": MERGE_KEY_MISSING, "value": 2}],
        tmp_path, "x", ("csv",), merge_keys=keys,
    )
    saved = pd.read_csv(tmp_path / "x.csv")
    assert len(saved) == 1
    assert saved["value"].iloc[0] == 2  # keep="last"
