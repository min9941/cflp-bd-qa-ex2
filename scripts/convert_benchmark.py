"""OR-Library CFLP 벤치마크를 본 프로젝트의 JSON 형식으로 변환한다.

사용법::

    # 파일 하나
    python scripts/convert_benchmark.py cap41.txt

    # 여러 개 / 폴더 / glob
    python scripts/convert_benchmark.py cap41.txt cap42.txt
    python scripts/convert_benchmark.py data/benchmark/raw_or_library
    python scripts/convert_benchmark.py "downloads/cap*.txt"

입력 경로가 폴더면 그 안의 `cap*.txt` 를 모두 변환한다.
출력은 기본적으로 `data/benchmark/` 에 `{m}x{n}_b{번호}.json` 으로 저장된다.

변환에 실패한 파일이 있어도 나머지는 계속 변환하고, 마지막에 실패 목록을 보고한다.
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path

import _bootstrap  # noqa: F401
from cflp_bd_qa_ex2.data.benchmark import (
    BenchmarkParseError,
    benchmark_to_instance,
    summarize,
)
from cflp_bd_qa_ex2.data.io import instance_to_dict, save_instance, validate_instance_payload
from cflp_bd_qa_ex2.logging_utils import log_fields, setup_logging


def expand_inputs(patterns: list[str]) -> list[Path]:
    """입력 인자를 실제 파일 목록으로 펼친다.

    파일 경로, 폴더, glob 패턴을 모두 받는다. 폴더면 그 안의 `cap*.txt` 를 찾는다.

    Returns:
        중복 없이 정렬된 파일 경로 목록.
    """
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.extend(sorted(path.glob("cap*.txt")))
            found.extend(sorted(path.glob("cap*.TXT")))
        elif path.is_file():
            found.append(path)
        else:
            matches = [Path(p) for p in globlib.glob(pattern)]
            found.extend(sorted(p for p in matches if p.is_file()))
    unique = sorted({p.resolve() for p in found})
    return unique


def main() -> None:
    """입력 파일을 모두 변환해 저장한다."""
    parser = argparse.ArgumentParser(description="OR-Library 벤치마크 변환")
    parser.add_argument("inputs", nargs="+", help="파일, 폴더, 또는 glob 패턴")
    parser.add_argument("--out", default="data/benchmark", help="출력 디렉터리")
    args = parser.parse_args()

    logger = setup_logging()
    paths = expand_inputs(args.inputs)
    if not paths:
        logger.error("변환할 파일을 찾지 못했다: %s", args.inputs)
        return

    out_dir = Path(args.out)
    converted, failed = [], []

    for path in paths:
        try:
            instance = benchmark_to_instance(path)
            # 저장 전에 schema 를 검증한다.
            validate_instance_payload(instance_to_dict(instance))
            saved = save_instance(instance, out_dir)
        except BenchmarkParseError as exc:
            failed.append((path.name, str(exc)))
            logger.error("%s 변환 실패: %s", path.name, exc)
            continue
        except Exception as exc:  # pragma: no cover - 예기치 못한 입력
            failed.append((path.name, f"{type(exc).__name__}: {exc}"))
            logger.error("%s 변환 실패: %s", path.name, exc)
            continue

        converted.append(saved)
        log_fields(logger, f"{path.name} -> {saved.name}", summarize(instance))

    logger.info("변환 완료 %d개 / 실패 %d개 -> %s", len(converted), len(failed), out_dir)
    for name, reason in failed:
        logger.warning("  실패: %s (%s)", name, reason)


if __name__ == "__main__":
    main()
