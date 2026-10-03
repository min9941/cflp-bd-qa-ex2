"""테스트 공통 fixture.

테스트는 exhaustive enumeration 이 빠르게 끝나도록 ``theta_bits`` 를 줄인
전용 설정을 사용한다. 이것은 **검증 전용 설정**이며 본 실험 설정과 분리된다.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cflp_bd_qa_ex2.config.loader import load_config  # noqa: E402
from cflp_bd_qa_ex2.data.generator import generate_instance  # noqa: E402


@pytest.fixture(scope="session")
def pilot_config() -> dict:
    """본 실험용 파일럿 설정."""
    return load_config(ROOT / "config" / "experiments" / "pilot.yaml")


@pytest.fixture(scope="session")
def test_config(pilot_config: dict) -> dict:
    """빠른 검증을 위해 theta_bits 를 줄인 테스트 전용 설정."""
    cfg = copy.deepcopy(pilot_config)
    cfg["encoding"]["theta"]["bits"] = 4
    cfg["sa"]["num_reads"] = 50
    cfg["sa"]["sweeps"] = 100
    cfg["embedding"]["timeout"] = 20
    cfg["embedding"]["seeds"] = [2024, 2025]
    cfg["validation"]["exhaustive_max_variables"] = 18
    return cfg


@pytest.fixture(scope="session")
def small_instance(pilot_config: dict):
    """4x12 seed=100 인스턴스."""
    return generate_instance("4x12", 4, 12, 100, pilot_config)


@pytest.fixture(scope="session")
def tiny_instance(pilot_config: dict):
    """exhaustive 검증용 초소형 인스턴스 (3 facilities, 4 customers)."""
    return generate_instance("3x4", 3, 4, 7, pilot_config)


def pytest_collection_modifyitems(config, items):
    """Gurobi 를 쓸 수 없는 환경에서는 해당 테스트를 skip 한다.

    라이선스가 없거나 사용자 불일치인 환경에서 실패로 표시되면
    실제 코드 오류와 구분되지 않는다.
    """
    from cflp_bd_qa_ex2.solvers.gurobi import check_gurobi_available

    available, status, message = check_gurobi_available()
    if available:
        return
    skip = pytest.mark.skip(reason=f"Gurobi 사용 불가: {status} / {message}")
    needs_gurobi = (
        "original_model", "subproblem_dual", "benders", "penalty",
        "results_schema", "scripts_behavior", "decoding", "qubo",
    )
    for item in items:
        if any(token in item.nodeid for token in needs_gurobi):
            item.add_marker(skip)
