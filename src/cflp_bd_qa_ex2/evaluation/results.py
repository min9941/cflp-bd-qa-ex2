"""결과 저장과 provenance.

CSV 와 Parquet 을 모두 지원한다. 기존 field 의 의미를 바꾸지 않는다.
결과 파일은 **덮어쓰지 않고 병합**한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_VERSION = "0.1.0"

MERGE_KEY_MISSING = "__NOT_APPLICABLE__"
"""merge key 에서 "해당 없음"을 나타내는 sentinel.

빈 문자열이나 ``NA`` / ``N/A`` / ``null`` 은 ``pd.read_csv`` 가 결측으로 읽으므로
저장-재로딩 왕복에서 값이 바뀐다. 그러면 같은 행이 중복 저장된다.
"""


def _package_version(name: str) -> str | None:
    """설치된 패키지 버전을 조회한다."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _decode(raw: bytes | None) -> str:
    """subprocess 출력 바이트를 OS 로캘과 무관하게 디코딩한다.

    Windows 한국어 환경의 기본 인코딩은 ``cp949`` 이므로,
    UTF-8 로 기록된 git 출력을 ``text=True`` 로 받으면 실패한다.
    디코딩할 수 없는 바이트는 대체 문자로 바꿔 provenance 수집이
    전체 실행을 중단시키지 않도록 한다.
    """
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
"""패키지 위치로부터 추정한 프로젝트 루트 (``src/`` 의 부모)."""


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    """프로젝트의 git commit 해시를 조회한다 (git 저장소가 아니면 None).

    현재 작업 디렉터리가 아니라 **프로젝트 경로** 기준으로 조회한다.
    notebook 이나 임의 위치에서 실행해도 같은 값이 나와야 하기 때문이다.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir or PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        # text=True 를 쓰면 Windows 한국어 로캘에서 cp949 로 디코딩을 시도해
        # UnicodeDecodeError 가 난다. 항상 바이트로 받아 UTF-8 로 명시 디코딩한다.
        return _decode(out.stdout).strip() or None
    except Exception:
        return None


def git_state(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """git commit 과 working tree 의 청결 여부를 함께 조회한다.

    ``git_dirty`` 가 True 이면 그 결과는 기록된 commit 의 **깨끗한 상태에서
    생성된 것이 아니다.** commit 만 기록하면 이 사실이 숨겨진다.

    Returns:
        ``git_commit``, ``git_dirty``, ``git_diff_hash`` 를 담은 dict.
        ``git_diff_hash`` 는 uncommitted 변경분(tracked diff + untracked 파일의
        경로·내용 해시)의 SHA-256 앞 16자리이며, 깨끗하면 ``None`` 이다.
    """
    repo = str(repo_dir or PROJECT_ROOT)
    commit = git_commit(repo)
    if commit is None:
        return {"git_commit": None, "git_dirty": None, "git_diff_hash": None}
    try:
        # git diff 출력에는 소스의 한국어 주석/docstring 이 UTF-8 로 들어간다.
        # Windows 한국어 로캘에서 text=True 로 받으면 cp949 디코딩에 실패한다.
        diff_proc = subprocess.run(
            ["git", "-C", repo, "diff", "HEAD"],
            capture_output=True, timeout=20, check=False,
        )
        # Git 은 기본적으로 비ASCII 경로를 "\355\225\234..." 로 이스케이프해 출력한다.
        # 그대로 읽으면 한글 파일명을 실제 경로로 찾지 못해 내용 변경을 놓친다.
        # core.quotepath=false 와 -z 를 함께 써서 **원시 바이트 경로**를 받는다.
        untracked_proc = subprocess.run(
            [
                "git", "-C", repo, "-c", "core.quotepath=false",
                "ls-files", "--others", "--exclude-standard", "-z",
            ],
            capture_output=True, timeout=20, check=False,
        )

        # returncode 를 확인하지 않고 stdout 만 쓰면, 명령이 실패해 출력이 비었을 때
        # "깨끗한 tree" 로 잘못 기록된다. 실패는 모름(None)으로 남긴다.
        if diff_proc.returncode != 0 or untracked_proc.returncode != 0:
            return {"git_commit": commit, "git_dirty": None, "git_diff_hash": None}

        diff = _decode(diff_proc.stdout)
        untracked_raw = untracked_proc.stdout
    except Exception:
        return {"git_commit": commit, "git_dirty": None, "git_diff_hash": None}

    # untracked 는 파일명만 나열되므로 내용 변경을 감지하지 못한다.
    # 경로와 내용 해시를 함께 넣어야 diff hash 가 상태를 온전히 식별한다.
    untracked_parts: list[str] = []
    raw_names = [chunk for chunk in untracked_raw.split(b"\0") if chunk]
    for raw in sorted(raw_names):
        # 파일 접근은 바이트 경로로 직접 한다 (디코딩 손실 방지).
        path = os.path.join(os.fsencode(repo), raw)
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            digest = "unreadable"
        name = os.fsdecode(raw)
        untracked_parts.append(f"{name}:{digest}")

    payload = diff + "\n".join(untracked_parts)
    dirty = bool(diff.strip()) or bool(untracked_parts)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_hash": (
            hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] if dirty else None
        ),
    }


def provenance(cfg: dict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """모든 실행에 붙일 provenance dict 를 만든다 (명세 27절)."""
    packages = {
        name: _package_version(name)
        for name in (
            "numpy",
            "pandas",
            "gurobipy",
            "dimod",
            "dwave-neal",
            "minorminer",
            "dwave-system",
            "dwave-networkx",
            "networkx",
        )
    }
    record = {
        "project_version": PROJECT_VERSION,
        **git_state(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": json.dumps(packages, sort_keys=True),
        "configuration_hash": cfg.get("configuration_hash"),
    }
    if extra:
        record.update(extra)
    return record


def save_records(
    records: Sequence[dict[str, Any]],
    out_dir: str | Path,
    name: str,
    formats: Sequence[str] = ("csv", "parquet"),
    merge_keys: Sequence[str] | None = None,
) -> list[Path]:
    """결과 레코드를 저장한다.

    ``merge_keys`` 가 주어지면 기존 파일을 읽어 해당 key 기준으로
    **병합(merge-not-overwrite)** 한다. 같은 key 의 새 행이 기존 행을 대체한다.

    Args:
        records: 저장할 레코드 목록.
        out_dir: 출력 디렉터리.
        name: 파일 이름(확장자 제외).
        formats: ``"csv"``, ``"parquet"`` 중 사용할 형식.
        merge_keys: 병합 기준 key. ``None`` 이면 단순 append 후 중복 제거 없음.

    Returns:
        저장된 파일 경로 목록.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(records))

    csv_path = out_dir / f"{name}.csv"
    if merge_keys and csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, frame], ignore_index=True)
        present = [k for k in merge_keys if k in combined.columns]
        if present:
            # merge key 의 결측을 정규화한다.
            # 빈 문자열로 저장한 값이 pd.read_csv 에서 NaN 으로 돌아오면
            # 같은 행인데도 key 가 달라져 중복 제거가 되지 않는다.
            for key in present:
                combined[key] = combined[key].where(combined[key].notna(), MERGE_KEY_MISSING)
                combined[key] = combined[key].replace("", MERGE_KEY_MISSING)
            combined = combined.drop_duplicates(subset=present, keep="last")
        frame = combined

    written: list[Path] = []
    if "csv" in formats:
        frame.to_csv(csv_path, index=False)
        written.append(csv_path)
    if "parquet" in formats:
        parquet_path = out_dir / f"{name}.parquet"
        try:
            frame.astype(object).where(pd.notnull(frame), None).to_parquet(
                parquet_path, index=False
            )
            written.append(parquet_path)
        except Exception:
            # parquet 엔진이 없거나 혼합 타입으로 실패해도 CSV 는 유지한다.
            pass
    return written


QA_MERGE_KEYS: dict[str, tuple[str, ...]] = {
    "qa_embedding_trials": (
        "instance_id", "trajectory_source", "benders_iteration",
        "topology_label", "graph_id", "embedding_seed", "configuration_hash",
    ),
    "qa_per_embedding": (
        "instance_id", "benders_iteration", "embedding_seed",
        "qa_repeat_id", "solver_id", "graph_id", "configuration_hash",
    ),
    "qa_iterations": (
        "instance_id", "trajectory_source", "benders_iteration",
        "graph_id", "configuration_hash",
    ),
    "qa_runs": (
        "instance_id", "trajectory_source", "graph_id", "configuration_hash",
    ),
}
"""QA 결과 파일별 merge key. checkpoint 저장과 최종 저장이 같은 key 를 쓴다."""


def save_qa_checkpoint(
    results_root: str | Path,
    topology: str,
    cfg: dict[str, Any],
    qa_embedding_rows: Sequence[dict[str, Any]],
    qa_per_embedding_rows: Sequence[dict[str, Any]],
    qa_iteration_rows: Sequence[dict[str, Any]],
    qa_run_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """지금까지 모인 QA 결과를 즉시 저장한다 (checkpoint).

    인스턴스 하나가 끝날 때마다 호출한다. 마지막에 한꺼번에 저장하면
    커널 재시작이나 인터럽트로 **모든 행이 소실된다.**

    merge key 에 ``graph_id`` 와 ``configuration_hash`` 가 있으므로
    같은 파일을 반복 저장해도 행이 중복되지 않는다.

    Args:
        results_root: ``results`` 디렉터리 경로.
        topology: ``"pegasus"`` 또는 ``"zephyr"``. 파일명 접미사로 쓰인다.
        cfg: 전체 설정 dict.
        qa_embedding_rows: QA 경로의 embedding trial(실패 포함).
        qa_per_embedding_rows: embedding seed 별 QA 평가.
        qa_iteration_rows: Benders iteration 기록.
        qa_run_rows: Benders run 요약.

    Returns:
        파일별 저장 행 수.
    """
    results_root = Path(results_root)
    formats = cfg["output"]["formats"]
    suffix = f"_{topology}"
    written: dict[str, int] = {}

    targets = (
        ("qa_embedding_trials", qa_embedding_rows, results_root / "embedding"),
        ("qa_per_embedding", qa_per_embedding_rows, results_root / "qa"),
        ("qa_iterations", qa_iteration_rows, results_root / "qa"),
        ("qa_runs", qa_run_rows, results_root / "qa"),
    )
    for name, rows, out_dir in targets:
        if not rows:
            written[f"{name}{suffix}"] = 0
            continue
        # 먼저 끝난 행의 provenance 를 나중 시점 값으로 덮어쓰지 않는다.
        save_records(
            attach_provenance_once(rows, cfg), out_dir, f"{name}{suffix}",
            formats, merge_keys=QA_MERGE_KEYS[name],
        )
        written[f"{name}{suffix}"] = len(rows)
    return written


EMBEDDING_MERGE_KEYS: tuple[str, ...] = (
    "instance_id", "trajectory_source", "benders_iteration",
    "topology_label", "graph_id", "embedding_seed", "configuration_hash",
)
"""controlled embedding trial 의 merge key."""


def save_embedding_checkpoint(
    results_root: str | Path,
    cfg: dict[str, Any],
    records: Sequence[dict[str, Any]],
    name: str = "embedding_trials",
) -> int:
    """controlled embedding trial 을 즉시 저장한다 (checkpoint).

    인스턴스 하나가 끝날 때마다 호출한다. 마지막에 한꺼번에 저장하면
    16x50 이나 25x50 도중 중단 시 **앞선 인스턴스 결과까지 소실된다.**
    성공한 mapping JSON 만 남고 trial 행은 남지 않는다.

    Returns:
        저장한 행 수.
    """
    if not records:
        return 0
    save_records(
        attach_provenance_once(records, cfg),
        Path(results_root) / "embedding",
        name,
        cfg["output"]["formats"],
        merge_keys=EMBEDDING_MERGE_KEYS,
    )
    return len(records)


def load_qa_results(results_dir: str | Path, stem: str) -> "pd.DataFrame":
    """topology 별로 나뉜 QA 결과를 읽어 합친다.

    ``qa_runs.csv`` (구형, 접미사 없음) 와 ``qa_runs_pegasus.csv`` 가 공존하면
    **같은 실행이 두 번 집계되어** 평균과 성공률이 왜곡된다.
    따라서 topology 접미사 파일이 하나라도 있으면 구형 파일은 무시한다.

    Args:
        results_dir: ``results/qa`` 경로.
        stem: ``"qa_runs"`` 같은 파일 이름 앞부분.

    Returns:
        합쳐진 DataFrame. 파일이 없으면 빈 DataFrame.
    """
    results_dir = Path(results_dir)
    legacy = results_dir / f"{stem}.csv"
    suffixed = sorted(p for p in results_dir.glob(f"{stem}_*.csv"))

    if suffixed:
        paths = suffixed
        if legacy.exists():
            print(
                f"  경고: 구형 파일 {legacy.name} 을 무시한다 "
                f"(topology 별 파일 {len(suffixed)}개 사용). 중복 집계 방지"
            )
    else:
        paths = [legacy] if legacy.exists() else []

    frames = [pd.read_csv(path) for path in paths]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_provenance(
    records: Sequence[dict[str, Any]], cfg: dict, extra: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """모든 레코드에 provenance field 를 붙인다 (기존 값을 덮어쓴다)."""
    prov = provenance(cfg, extra)
    return [{**record, **prov} for record in records]


PROVENANCE_MARKER = "timestamp"
"""provenance 가 이미 붙었는지 판정하는 field."""


def attach_provenance_once(
    records: Sequence[dict[str, Any]], cfg: dict, extra: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """provenance 가 **없는 레코드에만** 붙인다. **원본 dict 를 제자리에서 갱신한다.**

    checkpoint 저장은 누적 리스트 전체를 반복 저장한다. 매번
    :func:`attach_provenance` 를 쓰면 먼저 끝난 행의 ``timestamp``,
    ``git_commit``, ``git_dirty``, ``git_diff_hash``, ``package_versions`` 가
    나중 시점 값으로 교체되어, **행이 실제로 생성된 시점의 상태를 잃는다.**

    새 dict 를 반환하기만 하면 호출자의 누적 리스트는 여전히 provenance 가
    없는 상태로 남아, 다음 checkpoint 에서 또 새 값이 붙는다. 그래서 이 함수는
    **원본 record 를 직접 갱신한다**(side effect 가 의도된 동작이다).

    Args:
        records: 누적 레코드 리스트. provenance 가 없는 항목만 갱신된다.
        cfg: 전체 설정 dict.
        extra: 추가로 붙일 field.

    Returns:
        입력과 같은 dict 객체들의 리스트.
    """
    prov = provenance(cfg, extra)
    out: list[dict[str, Any]] = []
    for record in records:
        if record.get(PROVENANCE_MARKER) is None:
            record.update(prov)  # 제자리 갱신: 다음 checkpoint 에서 보존된다
        out.append(record)
    return out
