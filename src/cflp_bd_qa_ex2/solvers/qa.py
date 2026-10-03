"""D-Wave QA sampler wrapper.

원칙:

- **이상적 target graph 에서는 QA sampling 을 수행하지 않는다.**
  실제 working graph embedding 에서만 수행한다.
- 성공한 embedding seed 를 **모두** QA run 으로 연결한다.
  가장 짧은 chain 이나 가장 적은 physical qubit 의 embedding 만 고르지 않는다.
- QPU 는 재현 가능한 ``qa_seed`` 를 제공하지 않으므로 해당 field 를 쓰지 않는다.
  대신 ``embedding_seed``, ``qa_repeat_id``, ``qpu_problem_id``, ``solver_id``,
  ``graph_id``, ``timestamp`` 를 기록한다.
- Chain-break 처리는 ``majority_vote`` 로 고정하며 다른 방법과 비교하지 않는다.

Constraint ``penalty_alpha`` 와 chain strength 를 혼동하지 않는다.
chain strength 는 embedding 성공 후 ``uniform_torque_compensation`` 으로 계산한다.
``auto_scale`` 은 균일 스케일링일 뿐이며 coefficient dynamic range 를
개선하지 않는다.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any

from ..qubo.builder import MasterQUBO
from ..status import Status


QPU_TIMING_KEYS: tuple[str, ...] = (
    "qpu_sampling_time",
    "qpu_anneal_time_per_sample",
    "qpu_readout_time_per_sample",
    "qpu_access_time",
    "qpu_access_overhead_time",
    "qpu_programming_time",
    "qpu_delay_time_per_sample",
    "total_post_processing_time",
    "post_processing_overhead_time",
)
"""D-Wave 가 반환하는 표준 timing key.

모두 **마이크로초** 단위이다. 결과 schema 를 실행마다 흔들지 않기 위해,
해당 key 가 없어도 열은 항상 만들고 값만 ``None`` 으로 둔다.
"""


def classify_qpu_exception(exc: Exception) -> tuple[Status, str]:
    """QPU 제출/응답 예외를 status 로 분류한다.

    모든 예외를 ``NO_QPU_ACCESS`` 로 뭉뚱그리지 않는다.
    특히 **할당 시간 소진**은 접근 권한 없음과 구분해야 한다.
    권한이 없는 것이 아니라, 권한은 있는데 쓸 시간이 없는 상태이기 때문이다.

    Args:
        exc: 발생한 예외.

    Returns:
        ``(status, message)``.
    """
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()

    quota_hints = (
        "insufficient remaining solver access time",
        "quota",
        "exceeded",
        "no remaining",
    )
    if any(hint in lowered for hint in quota_hints):
        return Status.QPU_QUOTA_EXHAUSTED, f"{name}: {text}"

    if name in ("SolverAuthenticationError", "SolverOfflineError", "SolverNotFoundError"):
        return Status.NO_QPU_ACCESS, f"{name}: {text}"

    access_hints = ("auth", "token", "unauthorized", "forbidden", "connect",
                    "network", "dns", "ssl", "403", "401", "offline")
    if any(hint in lowered for hint in access_hints):
        return Status.NO_QPU_ACCESS, f"{name}: {text}"

    if "embed" in lowered or "chain" in lowered:
        return Status.EMBEDDING_INVALID, f"{name}: {text}"

    return Status.SOLVER_FAILED, f"{name}: {text}"


def flatten_qpu_timing(timing: dict[str, Any] | None) -> dict[str, Any]:
    """D-Wave timing dict 를 평탄한 결과 field 로 바꾼다.

    dict 를 그대로 저장하면 CSV 에서 문자열이 되어 집계할 때마다 파싱해야 한다.
    ``qa_<key>_us`` 형태의 개별 열로 펼친다. 단위는 원본 그대로 마이크로초이다.

    Args:
        timing: ``sampleset.info["timing"]``. ``None`` 이면 모든 값이 ``None`` 이다.

    Returns:
        ``qa_qpu_access_time_us`` 같은 key 를 갖는 dict.
        표준 key 는 항상 포함되고, 그 밖의 숫자 key 도 같은 규칙으로 추가된다.
    """
    timing = timing or {}
    flat: dict[str, Any] = {f"qa_{key}_us": None for key in QPU_TIMING_KEYS}
    for key, value in timing.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        flat[f"qa_{key}_us"] = float(value)
    return flat


def compute_chain_strength(qubo: MasterQUBO, embedding: dict, cfg: dict) -> dict[str, Any]:
    """``uniform_torque_compensation`` 으로 chain strength 를 계산한다.

    Args:
        qubo: 대상 QUBO.
        embedding: 검증을 통과한 embedding.
        cfg: 전체 설정 dict.

    Returns:
        ``chain_strength_method``, ``chain_strength_prefactor``,
        ``chain_strength`` 를 담은 dict.
    """
    from dwave.embedding.chain_strength import uniform_torque_compensation

    cs_cfg = cfg["qa"]["chain_strength"]
    prefactor = float(cs_cfg["prefactor"])
    bqm = qubo.to_bqm()
    value = float(
        uniform_torque_compensation(bqm, embedding=embedding, prefactor=prefactor)
    )
    return {
        "chain_strength_method": cs_cfg["method"],
        "chain_strength_prefactor": prefactor,
        "chain_strength": value,
    }


def run_qa(
    qubo: MasterQUBO,
    embedding: dict,
    solver_id: str,
    graph_id: str,
    cfg: dict,
    embedding_seed: int,
    qa_repeat_id: int = 0,
) -> dict[str, Any]:
    """실제 QPU 에 QUBO 를 제출한다.

    Args:
        qubo: 대상 QUBO.
        embedding: 실제 working graph 에서 검증된 embedding.
        solver_id: QPU solver id.
        graph_id: working graph 지문.
        cfg: 전체 설정 dict.
        embedding_seed: 이 embedding 을 만든 seed.
        qa_repeat_id: 같은 embedding 의 반복 실행 번호.

    Returns:
        ``samples`` 와 QA metadata 를 담은 dict. QPU 접근 실패 시
        ``status`` 가 ``NO_QPU_ACCESS`` 이고 ``samples`` 는 빈 리스트이다.
        **예외를 밖으로 던지지 않는다** (프로젝트 전체가 crash 하지 않도록).
    """
    qa_cfg = cfg["qa"]
    result: dict[str, Any] = {
        "status": str(Status.SUCCESS),
        "samples": [],
        # 실패해도 열 구성이 달라지지 않도록 timing 열을 미리 만들어 둔다.
        **flatten_qpu_timing(None),
        "qpu_runtime": None,
        "embedding_seed": embedding_seed,
        "qa_repeat_id": qa_repeat_id,
        "solver_id": solver_id,
        "graph_id": graph_id,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "qa_num_reads": int(qa_cfg["num_reads"]),
        "qa_annealing_time": float(qa_cfg["annealing_time"]),
        "qa_auto_scale": bool(qa_cfg["auto_scale"]),
        "chain_break_method": qa_cfg["chain_break_method"],
        "qa_profile": qa_cfg.get("profile"),
    }

    try:
        from dwave.system import DWaveSampler, FixedEmbeddingComposite
        from dwave.embedding.chain_breaks import majority_vote
    except Exception as exc:
        result["status"] = str(Status.NO_QPU_ACCESS)
        result["message"] = f"dwave-system import 실패: {exc}"
        return result

    try:
        result.update(compute_chain_strength(qubo, embedding, cfg))
    except Exception as exc:
        result["status"] = str(Status.SOLVER_FAILED)
        result["message"] = f"chain strength 계산 실패: {exc}"
        return result

    try:
        # profile 은 ~/.config/dwave/dwave.conf 의 프로필 이름이다.
        # **token 은 설정 파일에 넣지 않는다.** config snapshot 이 결과 폴더에
        # 저장되므로 credential 이 새어 나간다. token 은 DWAVE_API_TOKEN 환경변수로만 받는다.
        client_kwargs: dict[str, Any] = {"solver": solver_id}
        profile = qa_cfg.get("profile")
        if profile:
            client_kwargs["profile"] = profile
        base = DWaveSampler(**client_kwargs)
        sampler = FixedEmbeddingComposite(base, embedding)
        bqm = qubo.to_bqm()

        start = time.perf_counter()
        sampleset = sampler.sample(
            bqm,
            num_reads=int(qa_cfg["num_reads"]),
            annealing_time=float(qa_cfg["annealing_time"]),
            auto_scale=bool(qa_cfg["auto_scale"]),
            chain_strength=result["chain_strength"],
            chain_break_method=majority_vote,
            return_embedding=True,
        )
        # D-Wave 의 SampleSet 은 **비동기**이다. sample() 은 future 를 들고 바로 돌아오고,
        # 실제 오류(할당량 소진, 인증 실패 등)는 결과를 읽는 순간에 터진다.
        # resolve() 를 try 안에서 명시적으로 호출해야 예외를 여기서 분류할 수 있다.
        sampleset.resolve()
        info = dict(sampleset.info)
        wall_runtime = time.perf_counter() - start
    except Exception as exc:
        status, message = classify_qpu_exception(exc)
        result["status"] = str(status)
        result["message"] = f"QPU 제출/응답 실패: {message}"
        result["exception_type"] = type(exc).__name__
        return result

    result["qpu_problem_id"] = info.get("problem_id")
    timing = info.get("timing", {})

    # timing 은 마이크로초 단위 개별 열로 펼쳐 저장한다.
    result.update(flatten_qpu_timing(timing))

    # qpu_runtime 은 초 단위 요약값이다(Benders run-level 누적에 쓰인다).
    # 값이 없을 때 0.0 으로 채우면 "0초 걸렸다"와 "측정값이 없다"가 구분되지 않으므로
    # None 으로 둔다.
    access_time = timing.get("qpu_access_time")
    result["qpu_runtime"] = (
        float(access_time) / 1e6 if isinstance(access_time, (int, float)) else None
    )
    result["qa_wall_runtime"] = wall_runtime

    # chain-break 통계는 **aggregate 이전 원본 read** 에서 계산한다.
    # aggregate() 는 num_occurrences 만 누적하므로, 같은 decoded sample 이
    # 서로 다른 chain-break fraction 을 가졌을 때 평균이 왜곡된다.
    raw_cbf: list[float] = []
    if "chain_break_fraction" in sampleset.record.dtype.names:
        for row in sampleset.record:
            raw_cbf.extend([float(row.chain_break_fraction)] * int(row.num_occurrences))

    aggregated = sampleset.aggregate()
    ordered = aggregated.record[aggregated.record.energy.argsort()]
    variables = list(aggregated.variables)
    has_cbf = "chain_break_fraction" in aggregated.record.dtype.names

    samples: list[tuple[dict[str, int], float, int, float | None]] = []
    for row in ordered:
        assignment = {var: int(val) for var, val in zip(variables, row.sample)}
        # aggregate 후의 chain_break_fraction 은 대표값일 뿐이므로 참고용으로만 전달한다.
        cbf = float(row.chain_break_fraction) if has_cbf else None
        samples.append((assignment, float(row.energy), int(row.num_occurrences), cbf))

    result["samples"] = samples
    result["num_distinct_samples"] = len(samples)
    result["num_raw_reads"] = len(raw_cbf) if raw_cbf else None
    if raw_cbf:
        result["mean_chain_break_fraction"] = sum(raw_cbf) / len(raw_cbf)
        result["max_chain_break_fraction"] = max(raw_cbf)
        result["chain_break_source"] = "raw_reads_before_aggregate"
    return result
