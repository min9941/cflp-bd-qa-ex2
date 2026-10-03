"""Embedding 검증과 metric 테스트."""

from __future__ import annotations

import networkx as nx
import pytest

from cflp_bd_qa_ex2.embedding.embedder import chain_metrics, find_embedding_with_seed
from cflp_bd_qa_ex2.embedding.storage import embedding_filename, load_embedding, save_embedding
from cflp_bd_qa_ex2.embedding.targets import build_ideal_target, target_graph_fingerprint
from cflp_bd_qa_ex2.embedding.validation import validate_embedding
from cflp_bd_qa_ex2.qubo.builder import build_master_qubo
from cflp_bd_qa_ex2.qubo.diagnostics import logical_graph
from cflp_bd_qa_ex2.status import NA, Status


@pytest.fixture(scope="module")
def small_target():
    """검증 전용 작은 target graph (4x4 grid)."""
    graph = nx.grid_2d_graph(4, 4)
    return nx.convert_node_labels_to_integers(graph)


def test_rejects_missing_chain(small_target):
    """chain 이 없는 logical variable 을 검출한다."""
    source = nx.Graph()
    source.add_nodes_from(["a", "b", "c"])
    source.add_edge("a", "b")
    embedding = {"a": [0], "b": [1]}  # c 누락
    valid, reason = validate_embedding(embedding, source, small_target)
    assert not valid
    assert "chain 이 없는" in reason


def test_rejects_overlapping_chains(small_target):
    """chain 간 physical qubit 중복을 검출한다."""
    source = nx.Graph()
    source.add_edge("a", "b")
    embedding = {"a": [0, 1], "b": [1]}
    valid, reason = validate_embedding(embedding, source, small_target)
    assert not valid
    assert "중복" in reason


def test_rejects_qubit_outside_target(small_target):
    """target 에 없는 qubit 을 검출한다."""
    source = nx.Graph()
    source.add_edge("a", "b")
    embedding = {"a": [0], "b": [9999]}
    valid, reason = validate_embedding(embedding, source, small_target)
    assert not valid
    assert "없는 physical qubit" in reason


def test_rejects_disconnected_chain(small_target):
    """chain 내부가 끊겨 있으면 거부한다."""
    source = nx.Graph()
    source.add_edge("a", "b")
    embedding = {"a": [0, 15], "b": [1]}  # 0과 15는 인접하지 않음
    valid, reason = validate_embedding(embedding, source, small_target)
    assert not valid
    assert "연결" in reason


def test_rejects_missing_coupler(small_target):
    """logical edge 를 실현할 coupler 가 없으면 거부한다."""
    source = nx.Graph()
    source.add_edge("a", "b")
    embedding = {"a": [0], "b": [15]}
    valid, reason = validate_embedding(embedding, source, small_target)
    assert not valid
    assert "coupler" in reason


def test_accepts_valid_embedding(small_target):
    """유효한 embedding 을 통과시킨다."""
    source = nx.Graph()
    source.add_edge("a", "b")
    embedding = {"a": [0], "b": [1]}
    valid, reason = validate_embedding(embedding, source, small_target)
    assert valid and reason == ""


def test_isolated_variable_must_be_embedded(small_target):
    """quadratic edge 가 없는 isolated 변수도 chain 을 가져야 한다."""
    source = nx.Graph()
    source.add_edge("a", "b")
    source.add_node("isolated")
    embedding = {"a": [0], "b": [1]}
    valid, _ = validate_embedding(embedding, source, small_target)
    assert not valid

    embedding["isolated"] = [5]
    valid, _ = validate_embedding(embedding, source, small_target)
    assert valid


def test_chain_metrics():
    """chain metric 계산이 정의대로 동작한다."""
    metrics = chain_metrics({"a": [0, 1], "b": [2]}, logical_variables=2)
    assert metrics["physical_qubits"] == 3
    assert metrics["min_chain_length"] == 1
    assert metrics["max_chain_length"] == 2
    assert metrics["embedding_overhead"] == pytest.approx(1.5)


def test_ideal_pegasus_embedding_succeeds(small_instance, test_config):
    """이상적 P16 에서 초기 MP QUBO 가 임베딩된다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    source = logical_graph(qubo)
    target = build_ideal_target("pegasus", 16, "ideal_pegasus_p16")
    trial = find_embedding_with_seed(source, target, 2024, 30, 3)
    assert trial.success
    assert trial.embedding_status is Status.SUCCESS
    assert set(trial.embedding.keys()) == set(qubo.variable_labels)
    assert trial.fields["embedding_overhead"] >= 1.0


def test_failed_trial_uses_na(small_instance, test_config):
    """실패한 trial 의 chain metric 은 NA 이다."""
    qubo = build_master_qubo(small_instance, [], test_config)
    source = logical_graph(qubo)
    tiny = build_ideal_target("pegasus", 2, "tiny_pegasus")
    trial = find_embedding_with_seed(source, tiny, 2024, 5, 1)
    assert not trial.success
    assert trial.fields["physical_qubits"] == NA
    assert "search budget" in trial.message or trial.embedding_status is Status.EMBEDDING_INVALID


def test_target_fingerprint_detects_defects():
    """결함이 다르면 target 지문이 달라진다."""
    graph = nx.grid_2d_graph(3, 3)
    graph = nx.convert_node_labels_to_integers(graph)
    other = graph.copy()
    other.remove_node(0)
    assert target_graph_fingerprint(graph) != target_graph_fingerprint(other)


def test_embedding_filename_contains_required_fields():
    """파일명에 필수 식별자가 모두 들어간다."""
    name = embedding_filename(
        instance_id="4x12_s100",
        benders_iteration=3,
        topology_label="ideal_pegasus_p16",
        solver_id=None,
        graph_id="ideal_pegasus_16",
        embedding_seed=2024,
        timeout=300,
        tries=5,
        source_graph_fingerprint="n22e66_abc",
    )
    for token in ("4x12_s100", "it003", "ideal_pegasus_p16", "ideal", "seed2024",
                  "to300", "tr5", "n22e66_abc"):
        assert token in name


def test_save_embedding_does_not_overwrite(tmp_path):
    """같은 이름이 있으면 덮어쓰지 않고 새 파일을 만든다."""
    embedding = {"a": [0, 1]}
    first = save_embedding(embedding, tmp_path, {"k": "v"}, "same.json")
    second = save_embedding(embedding, tmp_path, {"k": "v"}, "same.json")
    assert first != second
    assert first.exists() and second.exists()
    loaded, metadata = load_embedding(first)
    assert loaded == {"a": [0, 1]}
    assert metadata == {"k": "v"}
