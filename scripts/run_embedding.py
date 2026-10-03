"""Structural controlled study: cut trajectory snapshot 을 여러 target 에 임베딩한다.

Gurobi-Benders 가 만든 동일한 cut trajectory 의 모든 iteration snapshot 을
이상적 P16 / Z12 와 (접근 가능하면) 실제 QPU working graph 에 임베딩한다.
각 snapshot 에 embedding seed 를 모두 적용하며, **실패한 trial 도 보존한다.**
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
from cflp_bd_qa_ex2.benders.cuts import build_optimality_cut
from cflp_bd_qa_ex2.benders.solver import GurobiMasterProvider, run_benders
from cflp_bd_qa_ex2.config.loader import load_config
from cflp_bd_qa_ex2.data.io import load_all_instances
from cflp_bd_qa_ex2.embedding.embedder import run_embedding_trials
from cflp_bd_qa_ex2.embedding.storage import embedding_filename, save_embedding
from cflp_bd_qa_ex2.embedding.targets import build_ideal_targets, select_actual_qpu
from cflp_bd_qa_ex2.evaluation.results import (
    attach_provenance,
    save_embedding_checkpoint,
    save_records,
)
from cflp_bd_qa_ex2.logging_utils import log_embedding, log_fields, setup_logging
from cflp_bd_qa_ex2.qubo.builder import build_master_qubo
from cflp_bd_qa_ex2.qubo.diagnostics import logical_graph, qubo_metrics, source_graph_fingerprint
from cflp_bd_qa_ex2.status import Status, TrajectorySource


def main() -> None:
    """Controlled trajectory 의 embedding metric 을 수집한다."""
    parser = argparse.ArgumentParser(description="Structural controlled embedding study")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--instances", default=None)
    parser.add_argument("--include-actual", action="store_true", help="실제 QPU working graph 도 포함")
    parser.add_argument(
        "--actual-only",
        action="store_true",
        help="ideal target 을 제외하고 실제 working graph 만 임베딩한다 (중복 실행 방지)",
    )
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    targets = [] if args.actual_only else build_ideal_targets(cfg)
    preflight_records: list[dict] = []
    if args.include_actual or args.actual_only:
        for key in cfg["qa"]["solvers"]:
            selection = select_actual_qpu(key, cfg)
            log_fields(
                logger,
                f"actual QPU ({key})",
                {"status": selection.status, "message": selection.message,
                 "available": selection.available_solvers},
            )
            # topology 별 선택 상태를 반드시 남긴다. 그렇지 않으면 결과 파일만 보고
            # "실행하지 않음"과 "NO_QPU_ACCESS"를 구분할 수 없다.
            record = {
                "record_type": "actual_embedding_preflight",
                "topology_key": key,
                "required_topology": cfg["qa"]["solvers"][key]["required_topology"],
                "required_shape": cfg["qa"]["solvers"][key]["required_shape"],
                "status": str(selection.status),
                "message": selection.message,
                "available_solvers": ", ".join(selection.available_solvers),
            }
            if selection.status is Status.SUCCESS and selection.target is not None:
                targets.append(selection.target)
                record.update(selection.target.as_record())
            preflight_records.append(record)

        save_records(
            attach_provenance(preflight_records, cfg),
            "results/embedding",
            "actual_embedding_preflight",
            cfg["output"]["formats"],
            merge_keys=("record_type", "topology_key", "configuration_hash"),
        )

    if not targets:
        logger.warning(
            "임베딩할 target 이 없다 (실제 QPU 미접근 + --actual-only). "
            "preflight 상태는 results/embedding/actual_embedding_preflight 에 저장했다"
        )
        return

    instances = load_all_instances(args.data)
    if args.instances:
        wanted = {s.strip() for s in args.instances.split(",")}
        instances = [i for i in instances if i.instance_id in wanted]

    records: list[dict] = []

    for inst in instances:
        trace = run_benders(inst, GurobiMasterProvider(), cfg)
        # cut snapshot 을 iteration 순서대로 재구성한다 (iteration k 의 QUBO 는 cut 0..k-1 을 갖는다).
        cuts = []
        snapshots = [list(cuts)]
        # **실제로 MP 에 추가된 cut 만** 사용한다.
        # 수렴 iteration 에서 생성되었으나 활성화되지 않은 마지막 cut 을 포함하면
        # 존재한 적 없는 MP 상태를 임베딩하게 된다.
        for snapshot in trace.cut_snapshots:
            if not snapshot.get("cut_activated", False):
                continue
            cuts = list(cuts)
            cuts.append(
                build_optimality_cut(
                    index=snapshot["cut_index"],
                    capacity=inst.capacity.astype(float),
                    u=np.asarray(snapshot["cut_dual_u"], dtype=float),
                    v=np.asarray(snapshot["cut_dual_v"], dtype=float),
                    iteration=snapshot["benders_iteration"],
                    dual_minimality_status=snapshot.get("dual_minimality_status", "NOT_CHECKED"),
                )
            )
            snapshots.append(list(cuts))

        for iteration, cut_set in enumerate(snapshots):
            qubo = build_master_qubo(inst, cut_set, cfg)
            source = logical_graph(qubo)
            fingerprint = source_graph_fingerprint(source)
            metrics = qubo_metrics(qubo)

            for target in targets:
                trials = run_embedding_trials(source, target, cfg)
                for trial in trials:
                    record = {
                        "instance_id": inst.instance_id,
                        "instance_name": inst.name,
                        "instance_seed": inst.seed,
                        "formulation": "ms_cflp_benders_mp",
                        "trajectory_source": str(TrajectorySource.GUROBI_CONTROLLED),
                        "benders_iteration": iteration,
                        "num_benders_cuts": len(cut_set),
                        "timeout": int(cfg["embedding"]["timeout"]),
                        "tries": int(cfg["embedding"]["tries"]),
                        "source_graph_fingerprint": fingerprint,
                        **target.as_record(),
                        **{
                            k: metrics[k]
                            for k in ("logical_variables", "logical_edges", "logical_density")
                        },
                        **qubo.constraint_metadata,
                        **trial.as_record(),
                    }
                    records.append(record)
                    log_embedding(logger, record)

                    if trial.success and trial.embedding is not None:
                        filename = embedding_filename(
                            instance_id=inst.instance_id,
                            benders_iteration=iteration,
                            topology_label=target.label,
                            solver_id=target.solver_id,
                            graph_id=target.graph_id,
                            embedding_seed=trial.embedding_seed,
                            timeout=int(cfg["embedding"]["timeout"]),
                            tries=int(cfg["embedding"]["tries"]),
                            source_graph_fingerprint=fingerprint,
                        )
                        saved = save_embedding(
                            trial.embedding,
                            "results/embedding/mappings",
                            {**record, "configuration_hash": cfg["configuration_hash"]},
                            filename,
                        )
                        record["embedding_file"] = str(saved)
                    else:
                        record["embedding_file"] = None

        # 인스턴스 하나가 끝날 때마다 저장한다.
        _flush(records, cfg)
        logger.info("%s 까지 %d행 저장", inst.instance_id, len(records))

    _flush(records, cfg)
    logger.info("embedding 결과 %d행을 results/embedding 에 저장했다", len(records))


def _flush(records: list[dict], cfg: dict) -> None:
    """지금까지 모은 embedding 결과를 저장한다 (중단되어도 결과가 남도록).

    provenance 는 행이 처음 저장될 때 한 번만 붙는다. 매번 새로 붙이면
    먼저 끝난 인스턴스의 timestamp 가 나중 시점 값으로 바뀐다.
    """
    if not records:
        return
    save_embedding_checkpoint("results", cfg, records)


