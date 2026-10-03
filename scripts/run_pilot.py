"""파일럿 전체 파이프라인을 순서대로 실행한다.

    generate_data -> run_gurobi -> run_sa
        -> run_embedding (ideal) -> run_embedding (actual)
        -> run_qa (pegasus) -> run_qa (zephyr)

각 단계는 독립적으로 실패할 수 있으며, 실패해도 다음 단계로 넘어간다.
QPU 접근 실패는 정상적인 결과(``NO_QPU_ACCESS``)로 취급한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> int:
    """하위 스크립트를 실행하고 return code 를 돌려준다."""
    command = [sys.executable, str(ROOT / script), *extra]
    print(f"\n===== {' '.join(command)} =====", flush=True)
    return subprocess.run(command, check=False).returncode


def main() -> None:
    """파일럿 전체를 실행한다."""
    parser = argparse.ArgumentParser(description="파일럿 전체 실행")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    parser.add_argument("--instances", default=None)
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument(
        "--skip-actual-embedding",
        action="store_true",
        help="실제 working graph embedding 을 건너뛴다 (기본은 포함)",
    )
    args = parser.parse_args()

    common = ["--config", args.config]
    scoped = list(common) + (["--instances", args.instances] if args.instances else [])

    results = {
        "generate_data": _run("generate_data.py", common),
        "run_gurobi": _run("run_gurobi.py", scoped + ["--encoded-mp"]),
        "run_sa": _run("run_sa.py", scoped),
        "run_embedding_ideal": _run("run_embedding.py", scoped),
    }
    if not args.skip_actual_embedding:
        # 실제 working graph 도 포함한다. QPU 접근 실패는 NO_QPU_ACCESS 로 기록될 뿐
        # 전체 파이프라인을 중단시키지 않는다.
        # ideal 은 위에서 이미 실행했으므로 --actual-only 로 중복 탐색을 피한다.
        results["run_embedding_actual"] = _run(
            "run_embedding.py", scoped + ["--actual-only"]
        )
    if not args.skip_qa:
        # Pegasus 와 Zephyr 를 모두 실행한다.
        for topology in ("pegasus", "zephyr"):
            results[f"run_qa_{topology}"] = _run(
                "run_qa.py", scoped + ["--topology", topology]
            )

    print("\n===== 단계별 return code =====")
    for name, code in results.items():
        print(f"  {name:16s} : {code}")


if __name__ == "__main__":
    main()
