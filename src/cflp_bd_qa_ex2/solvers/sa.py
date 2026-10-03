"""Simulated Annealing (dwave-neal) sampler wrapper.

SA seed 는 configurable 하며 결과에 기록된다.
Sample 은 energy 오름차순으로 검사한다(선택은 qubo/decode.py 가 담당).
"""

from __future__ import annotations

import time
from typing import Any

from ..qubo.builder import MasterQUBO
from ..status import Status


def run_sa(qubo: MasterQUBO, cfg: dict, context: dict | None = None) -> dict[str, Any]:
    """QUBO 를 SA 로 샘플링한다.

    Args:
        qubo: 대상 MP QUBO.
        cfg: 전체 설정 dict (``sa`` 하위 사용).
        context: 호출자가 넘기는 부가 정보(사용하지 않음, 인터페이스 통일용).

    Returns:
        ``samples`` (energy 오름차순 튜플 목록), ``sa_runtime``,
        ``sa_seed``, ``num_reads``, ``sweeps`` 를 담은 dict.
    """
    import neal

    sa_cfg = cfg["sa"]
    sampler = neal.SimulatedAnnealingSampler()
    bqm = qubo.to_bqm()

    start = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=int(sa_cfg["num_reads"]),
        num_sweeps=int(sa_cfg["sweeps"]),
        seed=int(sa_cfg["seed"]),
    )
    runtime = time.perf_counter() - start

    aggregated = sampleset.aggregate()
    ordered = aggregated.record[aggregated.record.energy.argsort()]
    variables = list(aggregated.variables)

    samples: list[tuple[dict[str, int], float, int, None]] = []
    for row in ordered:
        assignment = {var: int(val) for var, val in zip(variables, row.sample)}
        samples.append((assignment, float(row.energy), int(row.num_occurrences), None))

    return {
        "samples": samples,
        "status": str(Status.SUCCESS),
        "sa_runtime": runtime,
        "sa_seed": int(sa_cfg["seed"]),
        "sa_num_reads": int(sa_cfg["num_reads"]),
        "sa_sweeps": int(sa_cfg["sweeps"]),
        "num_distinct_samples": len(samples),
    }
