"""Terminal logging 유틸리티.

명세 28절이 요구하는 항목을 사람이 읽기 좋게 출력한다.
**토큰이나 credential 은 절대 출력하지 않는다.**
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_SECRET_HINTS = ("token", "secret", "password", "api_key", "apikey", "credential")

LOGGER_NAME = "cflp_bd_qa_ex2"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """표준 출력 logger 를 설정한다."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _is_secret(key: str) -> bool:
    """key 이름이 credential 로 보이는지 판정한다."""
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def log_fields(logger: logging.Logger, title: str, fields: dict[str, Any]) -> None:
    """key-value 묶음을 보기 좋게 출력한다 (credential 은 마스킹).

    Args:
        logger: 사용할 logger.
        title: 블록 제목.
        fields: 출력할 항목.
    """
    logger.info("--- %s ---", title)
    for key, value in fields.items():
        if _is_secret(key):
            logger.info("  %-34s : <masked>", key)
            continue
        if isinstance(value, float):
            logger.info("  %-34s : %.6g", key, value)
        else:
            logger.info("  %-34s : %s", key, value)


ITERATION_LOG_FIELDS: tuple[str, ...] = (
    "theta_lower_bound",
    "theta_upper_bound",
    "theta_delta",
    "theta_bits",
    "capacity_slack_bits",
    "cut_slack_bits_total",
    "penalty_alpha_capacity",
    "logical_variables",
    "logical_edges",
    "qubo_num_quadratic_terms",
    "qubo_coefficient_range",
    "energy",
    "decoded_mp_objective",
    "mp_feasible",
    "certified_lower_bound",
    "best_upper_bound",
)


def log_iteration(logger: logging.Logger, record: dict[str, Any]) -> None:
    """Benders iteration 기록 중 핵심 항목만 출력한다."""
    header = (
        f"[{record.get('instance_id')}] "
        f"seed={record.get('instance_seed')} "
        f"iter={record.get('benders_iteration')} "
        f"cuts={record.get('num_benders_cuts')} "
        f"source={record.get('trajectory_source')}"
    )
    logger.info(header)
    fields = {k: record[k] for k in ITERATION_LOG_FIELDS if k in record}
    if fields:
        log_fields(logger, "iteration", fields)


EMBEDDING_LOG_FIELDS: tuple[str, ...] = (
    "topology",
    "topology_label",
    "solver_id",
    "graph_id",
    "embedding_seed",
    "embedding_status",
    "physical_qubits",
    "mean_chain_length",
    "max_chain_length",
    "embedding_time",
)


def log_embedding(logger: logging.Logger, record: dict[str, Any]) -> None:
    """Embedding trial 기록 중 핵심 항목만 출력한다."""
    fields = {k: record[k] for k in EMBEDDING_LOG_FIELDS if k in record}
    log_fields(logger, "embedding", fields)
