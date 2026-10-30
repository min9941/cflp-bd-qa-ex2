"""OR-Library 벤치마크 변환 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from cflp_bd_qa_ex2.data.benchmark import (
    BenchmarkParseError,
    benchmark_to_instance,
    parse_or_library_cap,
)
from cflp_bd_qa_ex2.data.io import (
    instance_from_dict,
    instance_to_dict,
    validate_instance_payload,
)


def write_cap(tmp_path, name="cap99.txt", *, capacity=(100, 100), fixed=(50.0, 60.0),
              demand=(40, 30), total_cost=((400.0, 800.0), (90.0, 150.0))):
    """작은 cap 파일을 만든다.

    OR-Library 포맷: ``m n`` / 시설별 ``capacity fixed`` / 고객별 ``demand`` + 비용행.
    """
    lines = [f"{len(capacity)} {len(demand)}"]
    for s, f in zip(capacity, fixed):
        lines.append(f" {s} {f}")
    for d, row in zip(demand, total_cost):
        lines.append(f" {d}")
        lines.append(" " + " ".join(f"{v:.5f}" for v in row))
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_reads_all_tokens(tmp_path):
    """토큰을 빠짐없이 소비하고 shape 가 맞는다."""
    raw = parse_or_library_cap(write_cap(tmp_path))
    assert raw["n_facilities"] == 2 and raw["n_customers"] == 2
    assert raw["capacity"].tolist() == [100.0, 100.0]
    assert raw["total_cost"].shape == (2, 2)


def test_cost_is_converted_to_unit_cost(tmp_path):
    """OR-Library 의 총비용이 단위비용으로 변환되고 목적함수가 보존된다."""
    inst = benchmark_to_instance(write_cap(tmp_path))

    # total_cost[0] = (400, 800), demand[0] = 40 -> 단위비용 (10, 20)
    assert inst.transport_cost[0].tolist() == pytest.approx([10.0, 20.0])
    assert inst.transport_cost[1].tolist() == pytest.approx([3.0, 5.0])

    # d_i * unit_cost == 원본 총비용
    reconstructed = inst.transport_cost * inst.demand[:, None]
    assert reconstructed[0].tolist() == pytest.approx([400.0, 800.0])
    assert inst.metadata["unit_cost_roundtrip_max_error"] < 1e-9


def test_naming_follows_convention(tmp_path):
    """instance_id 는 {m}x{n}_b{번호}, name 은 cap{번호}."""
    inst = benchmark_to_instance(write_cap(tmp_path, "cap41.txt"))
    assert inst.instance_id == "2x2_b41"
    assert inst.name == "cap41"
    # 생성 데이터의 instance_name 과 겹치지 않는다
    assert inst.name not in {"4x12", "8x25", "16x50", "25x50"}


def test_locations_are_nan_and_flagged(tmp_path):
    """좌표는 NaN 이고 metadata 로 명시된다."""
    inst = benchmark_to_instance(write_cap(tmp_path))
    assert np.isnan(inst.facility_locations).all()
    assert np.isnan(inst.customer_locations).all()
    assert inst.metadata["has_locations"] is False


def test_json_roundtrip_writes_null_not_nan(tmp_path):
    """좌표가 JSON 에 null 로 기록되고 복원 시 NaN 이 된다."""
    import json

    inst = benchmark_to_instance(write_cap(tmp_path))
    payload = instance_to_dict(inst)
    assert payload["facility_locations"][0] == [None, None]

    # allow_nan=False 면 실제 NaN/Infinity 값이 하나라도 있을 때 예외가 난다.
    # (metadata 설명문에 "NaN" 이라는 **글자**가 들어 있으므로 문자열 검색은 부정확하다)
    text = json.dumps(payload, allow_nan=False)

    validate_instance_payload(payload)
    restored = instance_from_dict(json.loads(text))
    assert np.isnan(restored.facility_locations).all()
    assert np.array_equal(restored.demand, inst.demand)
    assert restored.transport_cost == pytest.approx(inst.transport_cost)


def test_demand_and_capacity_stay_integer(tmp_path):
    """capacity slack 을 delta=1 로 정확히 표현하려면 정수여야 한다."""
    inst = benchmark_to_instance(write_cap(tmp_path))
    assert inst.demand.dtype.kind == "i"
    assert inst.capacity.dtype.kind == "i"
    assert inst.capacity_slack_max == inst.total_capacity - inst.total_demand


def test_rejects_non_integer_demand(tmp_path):
    """정수가 아닌 demand 는 거부한다."""
    path = write_cap(tmp_path, demand=(40, 30))
    path.write_text(path.read_text(encoding="utf-8").replace("\n 30\n", "\n 30.5\n"),
                    encoding="utf-8")
    with pytest.raises(BenchmarkParseError, match="정수가 아니다"):
        benchmark_to_instance(path)


def test_rejects_insufficient_capacity(tmp_path):
    """총용량이 총수요보다 작으면 MP 가 infeasible 하므로 거부한다."""
    path = write_cap(tmp_path, capacity=(10, 10))
    with pytest.raises(BenchmarkParseError, match="총수요"):
        benchmark_to_instance(path)


def test_rejects_token_count_mismatch(tmp_path):
    """토큰 수가 맞지 않으면 조용히 통과시키지 않는다."""
    path = write_cap(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + " 123\n", encoding="utf-8")
    with pytest.raises(BenchmarkParseError, match="토큰 수 불일치"):
        benchmark_to_instance(path)


def test_rejects_bad_filename(tmp_path):
    """cap 형식이 아닌 파일명은 거부한다."""
    path = write_cap(tmp_path, "instance99.txt")
    with pytest.raises(BenchmarkParseError, match="벤치마크 번호"):
        benchmark_to_instance(path)


def test_expand_inputs_handles_file_dir_glob(tmp_path):
    """파일, 폴더, glob 입력을 모두 받는다."""
    import sys
    from pathlib import Path as _Path

    scripts = _Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from convert_benchmark import expand_inputs

    a = write_cap(tmp_path, "cap41.txt")
    b = write_cap(tmp_path, "cap42.txt")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert expand_inputs([str(a)]) == [a.resolve()]
    assert expand_inputs([str(tmp_path)]) == [a.resolve(), b.resolve()]
    assert expand_inputs([str(tmp_path / "cap4*.txt")]) == [a.resolve(), b.resolve()]
    assert expand_inputs([str(tmp_path / "missing*.txt")]) == []


def test_load_instances_merges_directories(tmp_path, pilot_config):
    """생성 데이터와 벤치마크를 함께 읽고, 없는 디렉터리는 건너뛴다."""
    from cflp_bd_qa_ex2.data.generator import generate_instance
    from cflp_bd_qa_ex2.data.io import load_instances, save_instance

    raw = tmp_path / "raw"
    bench = tmp_path / "benchmark"
    save_instance(generate_instance("4x12", 4, 12, 100, pilot_config), raw)
    save_instance(benchmark_to_instance(write_cap(tmp_path, "cap41.txt")), bench)

    merged = load_instances(raw, bench)
    assert {i.instance_id for i in merged} == {"4x12_s100", "2x2_b41"}
    assert {i.name for i in merged} == {"4x12", "cap41"}

    # 없는 디렉터리는 조용히 건너뛴다 (벤치마크 미변환 상태를 허용)
    assert len(load_instances(raw, tmp_path / "missing")) == 1
    with pytest.raises(FileNotFoundError):
        load_instances(raw, tmp_path / "missing", required=True)


def test_load_instances_rejects_duplicate_ids(tmp_path, pilot_config):
    """같은 instance_id 가 두 디렉터리에 있으면 임의로 고르지 않는다."""
    from cflp_bd_qa_ex2.data.generator import generate_instance
    from cflp_bd_qa_ex2.data.io import load_instances, save_instance

    first, second = tmp_path / "a", tmp_path / "b"
    instance = generate_instance("4x12", 4, 12, 100, pilot_config)
    save_instance(instance, first)
    save_instance(instance, second)

    with pytest.raises(ValueError, match="중복된 instance_id"):
        load_instances(first, second)


def test_benchmark_selection_requires_correct_seed_filter(tmp_path, pilot_config):
    """벤치마크는 seed 가 -1 이므로 생성 데이터의 seed 로는 선택되지 않는다."""
    from cflp_bd_qa_ex2.data.generator import generate_instance
    from cflp_bd_qa_ex2.evaluation.target_selection import (
        BENCHMARK_SEED,
        describe_selection,
        pick_instances,
    )

    instances = [
        generate_instance("25x50", 25, 50, 400, pilot_config),
        benchmark_to_instance(write_cap(tmp_path, "cap41.txt")),
    ]
    assert instances[1].seed == BENCHMARK_SEED

    # 흔한 실수: 벤치마크에 생성 데이터 seed 를 적용
    wrong = describe_selection(instances, ["cap41"], [400], "benchmark")
    assert wrong["instance_ids"] == []
    assert wrong["warnings"]  # 조용히 비지 않고 경고한다

    assert pick_instances(instances, ["cap41"], None) == ["2x2_b41"]
    assert pick_instances(instances, ["cap41"], [BENCHMARK_SEED]) == ["2x2_b41"]
    assert pick_instances(instances, ["25x50"], [400]) == ["25x50_s400"]
