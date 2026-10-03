"""파일럿 인스턴스를 생성하고 저장한다.

사용법::

    python scripts/generate_data.py --config config/experiments/pilot.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cflp_bd_qa_ex2.config.loader import load_config, snapshot_config
from cflp_bd_qa_ex2.data.generator import generate_all
from cflp_bd_qa_ex2.data.io import save_instance
from cflp_bd_qa_ex2.logging_utils import log_fields, setup_logging


def main() -> None:
    """인스턴스를 생성해 ``data/raw`` 에 저장한다."""
    parser = argparse.ArgumentParser(description="파일럿 인스턴스 생성")
    parser.add_argument("--config", default="config/experiments/pilot.yaml")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    snapshot_config(cfg, "results")

    instances = generate_all(cfg)
    out_dir = Path(args.out)
    for inst in instances:
        save_instance(inst, out_dir)
        log_fields(
            logger,
            inst.instance_id,
            {
                "n_facilities": inst.n_facilities,
                "n_customers": inst.n_customers,
                "total_demand": inst.total_demand,
                "total_capacity": inst.total_capacity,
                "capacity_slack_max": inst.capacity_slack_max,
                "objective_upper_bound": inst.objective_upper_bound,
            },
        )
    logger.info("총 %d개 인스턴스를 %s 에 저장했다", len(instances), out_dir)


if __name__ == "__main__":
    main()
