"""QA 실행: QPU preflight, controlled embedding, end-to-end QA Benders.

QPU 접근이 불가능해도 **전체 프로젝트가 crash 하지 않는다.**
접근 실패는 ``NO_QPU_ACCESS`` 로 기록하고 종료한다.

이상적 target graph 에서는 QA sampling 을 하지 않는다.
실제 working graph 에서 성공한 embedding seed 는 **모두** QA run 으로 연결한다.
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import _bootstrap  # noqa: F401
from cflp_bd_qa_ex2.benders.solver import QuboMasterProvider, run_benders
from cflp_bd_qa_ex2.config.loader import load_config
from cflp_bd_qa_ex2.data.io import load_all_instances
from cflp_bd_qa_ex2.embedding.embedder import run_embedding_trials
from cflp_bd_qa_ex2.embedding.storage import embedding_filename, save_embedding
from cflp_bd_qa_ex2.embedding.targets import list_accessible_qpus, select_actual_qpu
from cflp_bd_qa_ex2.evaluation.results import (
    attach_provenance,
    save_qa_checkpoint,
    save_records,
)
from cflp_bd_qa_ex2.logging_utils import log_fields, log_iteration, setup_logging
from cflp_bd_qa_ex2.qubo.decode import feasible_sample_rate, select_feasible_sample
from cflp_bd_qa_ex2.qubo.diagnostics import logical_graph, source_graph_fingerprint
from cflp_bd_qa_ex2.solvers.qa_master import QAMasterProblemSolver
from cflp_bd_qa_ex2.status import Status, TrajectorySource


def main() -> None:
    """QA preflight 후 end-to-end QA Benders 를 실행한다."""
    parser = argparse.ArgumentParser(description="QA end-to-end Benders")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--instances", default=None)
    parser.add_argument("--topology", default="pegasus", choices=["pegasus", "zephyr"])
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    status, solvers, message = list_accessible_qpus()
    log_fields(logger, "접근 가능한 QPU", {"status": status, "message": message})
    for info in solvers:
        log_fields(logger, info["solver_id"], info)

    selection = select_actual_qpu(args.topology, cfg)
    log_fields(
        logger,
        f"선택된 QPU ({args.topology})",
        {"status": selection.status, "message": selection.message,
         "available": selection.available_solvers},
    )

    records: list[dict] = []
    if selection.status is not Status.SUCCESS or selection.target is None:
        logger.warning("QPU 에 접근할 수 없어 QA 실행을 건너뛴다: %s", selection.status)
        records.append(
            {
                "record_type": "qa_preflight",
                "topology": args.topology,
                "status": str(selection.status),
                "message": selection.message,
                "available_solvers": ",".join(selection.available_solvers),
            }
        )
        save_records(
            attach_provenance(records, cfg), "results/qa", f"qa_preflight_{args.topology}",
            cfg["output"]["formats"],
            merge_keys=("record_type", "topology", "configuration_hash"),
        )
        return

    target = selection.target
    log_fields(logger, "working graph", target.as_record())

    instances = load_all_instances(args.data)
    if args.instances:
        wanted = {s.strip() for s in args.instances.split(",")}
        instances = [i for i in instances if i.instance_id in wanted]

    iteration_records, run_records, qa_records, embedding_records = [], [], [], []
    for inst in instances:
        sampler = QAMasterProblemSolver(
            target=target,
            instance_id=inst.instance_id,
            cfg=cfg,
            qa_records=qa_records,
            embedding_records=embedding_records,
        )

        provider = QuboMasterProvider(
            source=TrajectorySource.QA_END_TO_END,
            sampler=sampler,
        )
        trace = run_benders(inst, provider, cfg)
        for record in (it.as_record() for it in trace.iterations):
            record.update(target.as_record())
            iteration_records.append(record)
            log_iteration(logger, record)
        if trace.run is not None:
            record = trace.run.as_record()
            record.update(target.as_record())
            run_records.append(record)

        # 인스턴스 하나가 끝날 때마다 저장한다.
        # 마지막에 한 번만 저장하면 중단 시 모든 행이 소실된다.
        written = save_qa_checkpoint(
            "results", args.topology, cfg,
            embedding_records, qa_records, iteration_records, run_records,
        )
        logger.info("%s checkpoint 저장: %s", inst.instance_id, written)

    # notebook 과 같은 topology 접미사를 쓴다.
    # 접미사 없는 파일과 접미사 파일이 공존하면 06 에서 같은 실행이 두 번 집계된다.
    # 최종 저장도 checkpoint 와 **같은 함수**를 쓴다.
    # attach_provenance() 를 다시 쓰면 checkpoint 에서 보존한 provenance 를
    # 마지막에 전부 덮어쓴다.
    written = save_qa_checkpoint(
        "results", args.topology, cfg,
        embedding_records, qa_records, iteration_records, run_records,
    )
    logger.info("QA 결과를 저장했다: %s", written)


if __name__ == "__main__":
    main()
