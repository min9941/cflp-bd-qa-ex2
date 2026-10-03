"""Original CFLP ground truth 와 Gurobi-controlled Benders 를 실행한다.

제한(size-limited) 라이선스에서 모델이 너무 크면 ``GUROBI_SIZE_LIMIT`` 로
기록하고 해당 인스턴스만 건너뛴다. 전체 실행은 중단되지 않는다.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cflp_bd_qa_ex2.benders.solver import GurobiMasterProvider, run_benders
from cflp_bd_qa_ex2.config.loader import load_config
from cflp_bd_qa_ex2.data.io import load_all_instances
from cflp_bd_qa_ex2.evaluation.results import attach_provenance, save_records
from cflp_bd_qa_ex2.logging_utils import log_fields, setup_logging
from cflp_bd_qa_ex2.models.cflp import solve_original_cflp
from cflp_bd_qa_ex2.solvers.gurobi import check_gurobi_available, probe_size_limits
from cflp_bd_qa_ex2.status import Status


def main() -> None:
    """Ground truth 와 controlled Benders trajectory 를 저장한다."""
    parser = argparse.ArgumentParser(description="Gurobi ground truth + Benders")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--instances", default=None, help="쉼표로 구분한 instance_id 목록")
    parser.add_argument("--encoded-mp", action="store_true", help="매 iteration encoded MP 도 해결")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    available, status, message = check_gurobi_available()
    log_fields(logger, "Gurobi preflight", {"available": available, "status": status, "message": message})
    if not available:
        logger.error("Gurobi 를 사용할 수 없어 중단한다: %s", message)
        return
    log_fields(logger, "라이선스 크기 한계", probe_size_limits())

    instances = load_all_instances(args.data)
    if args.instances:
        wanted = {s.strip() for s in args.instances.split(",")}
        instances = [i for i in instances if i.instance_id in wanted]

    gt_records, iteration_records, run_records, cut_records = [], [], [], []

    for inst in instances:
        gt = solve_original_cflp(inst, cfg)
        gt_records.append(
            {
                "instance_id": inst.instance_id,
                "instance_name": inst.name,
                "instance_seed": inst.seed,
                "ground_truth_status": str(gt.status),
                "objective_opt": gt.objective if gt.status is Status.OPTIMAL else None,
                "incumbent_objective": gt.objective,
                "objective_bound": gt.bound,
                "runtime": gt.runtime,
                "message": gt.message,
                "ground_truth_y": gt.payload.get("y"),
            }
        )
        log_fields(
            logger,
            f"{inst.instance_id} original CFLP",
            {"status": gt.status, "objective": gt.objective, "runtime": gt.runtime},
        )

        if gt.status in (Status.GUROBI_SIZE_LIMIT, Status.GUROBI_LICENSE_UNAVAILABLE):
            logger.warning(
                "%s: 제한 라이선스로 original CFLP 를 풀 수 없다. Benders 도 SP 에서 실패한다",
                inst.instance_id,
            )

        trace = run_benders(
            inst, GurobiMasterProvider(), cfg, solve_encoded_each_iteration=args.encoded_mp
        )
        iteration_records.extend(it.as_record() for it in trace.iterations)
        if trace.run is not None:
            record = trace.run.as_record()
            if gt.status is Status.OPTIMAL and record.get("best_upper_bound") is not None:
                record["ground_truth_objective"] = gt.objective
                record["benders_vs_original_abs_diff"] = abs(
                    record["best_upper_bound"] - gt.objective
                )
            run_records.append(record)
            log_fields(
                logger,
                f"{inst.instance_id} Benders(gurobi_controlled)",
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
    save_records(attach_provenance(gt_records, cfg), "results/ground_truth", "ground_truth", formats,
                 merge_keys=("instance_id", "configuration_hash"))
    save_records(attach_provenance(iteration_records, cfg), "results/benders", "benders_iterations", formats,
                 merge_keys=("instance_id", "trajectory_source", "benders_iteration", "configuration_hash"))
    save_records(attach_provenance(run_records, cfg), "results/benders", "benders_runs", formats,
                 merge_keys=("instance_id", "trajectory_source", "configuration_hash"))
    save_records(attach_provenance(cut_records, cfg), "results/benders", "cut_trajectory", formats,
                 merge_keys=("instance_id", "trajectory_source", "benders_iteration", "cut_index", "configuration_hash"))
    logger.info("결과를 results/ 에 저장했다")


if __name__ == "__main__":
    main()
