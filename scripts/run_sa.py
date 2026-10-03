"""SA end-to-end Benders 를 실행한다.

SA 가 만든 candidate y 로 SP 를 풀고 cut 을 추가하므로,
cut trajectory 가 Gurobi-controlled 와 달라질 수 있다.
두 결과를 합치지 않는다.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cflp_bd_qa_ex2.benders.solver import QuboMasterProvider, run_benders
from cflp_bd_qa_ex2.config.loader import load_config, snapshot_config
from cflp_bd_qa_ex2.data.io import load_all_instances
from cflp_bd_qa_ex2.evaluation.results import attach_provenance, save_records
from cflp_bd_qa_ex2.logging_utils import log_fields, log_iteration, setup_logging
from cflp_bd_qa_ex2.solvers.sa import run_sa
from cflp_bd_qa_ex2.status import TrajectorySource


def main() -> None:
    """SA end-to-end Benders 결과를 저장한다."""
    parser = argparse.ArgumentParser(description="SA end-to-end Benders")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--instances", default=None, help="쉼표로 구분한 instance_id 목록")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=(
            "benders.max_iterations 를 덮어쓴다. 기본값은 config 의 1000 이며, "
            "덮어쓰면 configuration_hash 가 바뀌고 경고가 남는다. "
            "본 파일럿의 기준 설정은 SA 에도 1000 을 그대로 적용한다."
        ),
    )
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    if args.max_iterations is not None:
        from cflp_bd_qa_ex2.config.loader import config_hash

        cfg["benders"]["max_iterations"] = int(args.max_iterations)
        cfg["configuration_hash"] = config_hash(cfg)
        # 변경된 설정의 snapshot 을 반드시 남긴다.
        # 그렇지 않으면 결과의 hash 에 대응하는 설정 파일이 존재하지 않는다.
        snapshot = snapshot_config(cfg, "results")
        logger.warning(
            "benders.max_iterations 를 %d 로 덮어썼다. configuration_hash=%s, snapshot=%s",
            args.max_iterations,
            cfg["configuration_hash"][:16],
            snapshot,
        )

    instances = load_all_instances(args.data)
    if args.instances:
        wanted = {s.strip() for s in args.instances.split(",")}
        instances = [i for i in instances if i.instance_id in wanted]

    iteration_records, run_records, qubo_records, cut_records = [], [], [], []

    for inst in instances:
        collected: list[dict] = []
        provider = QuboMasterProvider(
            source=TrajectorySource.SA_END_TO_END,
            sampler=run_sa,
            collect_qubo_records=collected,
        )
        trace = run_benders(inst, provider, cfg)

        for record in (it.as_record() for it in trace.iterations):
            record["sa_seed"] = int(cfg["sa"]["seed"])
            record["sa_num_reads"] = int(cfg["sa"]["num_reads"])
            record["sa_sweeps"] = int(cfg["sa"]["sweeps"])
            iteration_records.append(record)
            log_iteration(logger, record)

        for idx, record in enumerate(collected, start=1):
            qubo_records.append(
                {
                    "instance_id": inst.instance_id,
                    "instance_name": inst.name,
                    "instance_seed": inst.seed,
                    "trajectory_source": str(TrajectorySource.SA_END_TO_END),
                    "benders_iteration": idx,
                    **record,
                }
            )

        if trace.run is not None:
            record = trace.run.as_record()
            record["sa_seed"] = int(cfg["sa"]["seed"])
            record["sa_num_reads"] = int(cfg["sa"]["num_reads"])
            record["sa_sweeps"] = int(cfg["sa"]["sweeps"])
            record["benders_max_iterations"] = int(cfg["benders"]["max_iterations"])
            run_records.append(record)
            log_fields(
                logger,
                f"{inst.instance_id} Benders(sa_end_to_end)",
                {
                    k: record.get(k)
                    for k in (
                        "termination_status",
                        "benders_iterations",
                        "certified_lower_bound",
                        "best_upper_bound",
                        "final_optimality_gap",
                        "dual_minimality_status",
                        "dual_guard_action",
                    )
                },
            )
        cut_records.extend(trace.cut_snapshots)

    formats = cfg["output"]["formats"]
    save_records(attach_provenance(iteration_records, cfg), "results/sa", "sa_iterations", formats,
                 merge_keys=("instance_id", "trajectory_source", "benders_iteration",
                             "configuration_hash", "sa_seed"))
    save_records(attach_provenance(run_records, cfg), "results/sa", "sa_runs", formats,
                 merge_keys=("instance_id", "trajectory_source", "configuration_hash", "sa_seed"))
    save_records(attach_provenance(qubo_records, cfg), "results/qubo", "qubo_metrics", formats,
                 merge_keys=("instance_id", "trajectory_source", "benders_iteration",
                             "configuration_hash"))
    save_records(attach_provenance(cut_records, cfg), "results/sa", "sa_cut_trajectory", formats,
                 merge_keys=("instance_id", "trajectory_source", "benders_iteration",
                             "cut_index", "configuration_hash"))
    logger.info("SA 결과를 results/sa 에 저장했다")


if __name__ == "__main__":
    main()
