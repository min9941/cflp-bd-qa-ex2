"""QA end-to-end 실험의 Master Problem solver.

``QAMasterProblemSolver`` 는 Ocean 의 sampler 가 아니다.
``dimod.Sampler`` 를 상속하지 않고 ``.sample(bqm)`` 도 제공하지 않는다.
Benders 루프가 매 iteration 호출하는 **MP 후보 생성기**이며, 계층은 다음과 같다.

    QAMasterProblemSolver   (이 모듈)      실험 오케스트레이션
            |
            +-- run_qa()    (solvers/qa.py)  D-Wave 호출 래퍼
                    |
                    +-- DWaveSampler / FixedEmbeddingComposite   실제 QPU 제출


명세 24절은 notebook 에 핵심 algorithm 을 구현하지 말라고 요구한다.
따라서 QA sampler 로직은 이 모듈에 두고 ``scripts/run_qa.py`` 와
``notebooks/05_1_run_qa_pegasus.ipynb`` 와 ``notebooks/05_2_run_qa_zephyr.ipynb`` 가
**같은 클래스**를 호출한다.

실험 설계는 **선택 A** 이다.

- 동일 QUBO 에서 성공한 모든 embedding seed 를 독립적으로 decode·평가한다
- sample 을 하나의 pool 로 합치지 않는다
- Benders trajectory 는 설정된 ``embedding.seeds`` 순서에서 성공한
  첫 seed 로 고정한다 ("가장 좋은 것"을 고르지 않는다)

seed 마다 완전히 독립된 end-to-end trajectory 를 돌리는 방식(선택 B)은
별도 실험으로 분리해야 하며 본 파일럿 범위가 아니다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..embedding.embedder import run_embedding_trials
from ..embedding.storage import embedding_filename, save_embedding
from ..embedding.targets import TargetGraph
from ..qubo.builder import MasterQUBO
from ..qubo.decode import feasible_sample_rate, select_feasible_sample
from ..qubo.diagnostics import logical_graph, source_graph_fingerprint
from ..logging_utils import LOGGER_NAME
from ..status import Status, TrajectorySource
from .qa import run_qa


@dataclass
class QAMasterProblemSolver:
    """QA end-to-end Benders 의 Master Problem solver.

    Ocean sampler 가 아니다. embedding 탐색 -> seed 별 QA 실행 -> 독립 평가 ->
    trajectory 후보 선택까지를 담당하며, 실제 QPU 제출은 ``run_qa()`` 가 한다.

    Attributes:
        target: 실제 working graph (ideal target 에서는 QA 를 수행하지 않는다).
        instance_id: 대상 인스턴스 식별자.
        cfg: 전체 설정 dict.
        mapping_dir: 성공한 embedding mapping 을 저장할 디렉터리.
        qa_records: embedding seed 별 QA 평가 결과가 누적되는 리스트.
        embedding_records: embedding trial(실패 포함)이 누적되는 리스트.
        verbose: True 면 iteration 별 진행 상황을 logger 로 출력한다.
            end-to-end QA 는 한 iteration 이 수 분 걸릴 수 있으므로 기본값은 True 이다.
    """

    target: TargetGraph
    instance_id: str
    cfg: dict
    mapping_dir: str = "results/embedding/qa_mappings"
    qa_records: list[dict[str, Any]] = field(default_factory=list)
    embedding_records: list[dict[str, Any]] = field(default_factory=list)
    verbose: bool = True

    def __call__(self, qubo: MasterQUBO, cfg: dict, context: dict) -> dict[str, Any]:
        """QUBO 를 임베딩하고 성공한 seed 를 각각 독립 실행·평가한다.

        Args:
            qubo: 현재 iteration 의 MP QUBO.
            cfg: 전체 설정 dict.
            context: ``instance``/``cuts``/``iteration`` 을 담은 dict.

        Returns:
            Benders trajectory 에 쓸 sample 과 metadata.
            embedding 이 모두 실패하면 ``status`` 가 ``EMBEDDING_FAILED`` 이고,
            QPU 제출이 실패하면 그 status 를 그대로 전달한다.
        """
        instance = context["instance"]
        cuts = context["cuts"]
        iteration = context["iteration"]
        target = self.target

        source = logical_graph(qubo)
        fingerprint = source_graph_fingerprint(source)

        logger = logging.getLogger(LOGGER_NAME)
        if self.verbose:
            logger.info(
                "[%s / %s] iter %d 시작: cuts=%d, logical vars=%d, edges=%d -> embedding 탐색 %d seed",
                self.instance_id, target.label, iteration, len(cuts),
                qubo.num_variables, len(qubo.quadratic), len(cfg["embedding"]["seeds"]),
            )

        start = time.perf_counter()
        trials = run_embedding_trials(source, target, cfg)
        embedding_runtime = time.perf_counter() - start

        # --- embedding trial 전량 보존 (실패 포함) + mapping 저장 ---
        embedding_files: dict[int, str | None] = {}
        for trial in trials:
            record = {
                "instance_id": self.instance_id,
                "trajectory_source": str(TrajectorySource.QA_END_TO_END),
                "benders_iteration": iteration,
                "num_benders_cuts": len(cuts),
                "timeout": int(cfg["embedding"]["timeout"]),
                "tries": int(cfg["embedding"]["tries"]),
                "source_graph_fingerprint": fingerprint,
                "logical_variables": qubo.num_variables,
                "logical_edges": len(qubo.quadratic),
                "qa_requested": True,
                "used_for_qa": bool(trial.success),
                **target.as_record(),
                **trial.as_record(),
            }
            if trial.success and trial.embedding is not None:
                saved = save_embedding(
                    trial.embedding,
                    self.mapping_dir,
                    {**record, "configuration_hash": cfg.get("configuration_hash")},
                    embedding_filename(
                        self.instance_id, iteration, target.label, target.solver_id,
                        target.graph_id, trial.embedding_seed,
                        int(cfg["embedding"]["timeout"]), int(cfg["embedding"]["tries"]),
                        fingerprint,
                    ),
                )
                record["embedding_file"] = str(saved)
                embedding_files[int(trial.embedding_seed)] = str(saved)
            else:
                record["embedding_file"] = None
            self.embedding_records.append(record)

        n_success = sum(1 for t in trials if t.success)
        if self.verbose:
            logger.info(
                "[%s / %s] iter %d embedding: 성공 %d/%d, %.1f초",
                self.instance_id, target.label, iteration,
                n_success, len(trials), embedding_runtime,
            )

        order = {int(seed): rank for rank, seed in enumerate(cfg["embedding"]["seeds"])}
        successes = sorted(
            (t for t in trials if t.success and t.embedding is not None),
            key=lambda t: order.get(int(t.embedding_seed), 1 << 30),
        )

        base = {
            "num_embedding_trials": len(trials),
            "num_successful_embeddings": len(successes),
            "embedding_runtime": embedding_runtime,
            "source_graph_fingerprint": fingerprint,
        }

        if not successes:
            if self.verbose:
                logger.warning(
                    "[%s / %s] iter %d: 주어진 search budget 에서 embedding 을 찾지 못함",
                    self.instance_id, target.label, iteration,
                )
            return {
                "samples": [],
                "status": str(Status.EMBEDDING_FAILED),
                "message": "주어진 search budget 에서 embedding 을 찾지 못함",
                **base,
            }

        chosen: tuple[Any, dict[str, Any]] | None = None
        qpu_runtime_total = 0.0
        qpu_calls = 0

        for trial in successes:
            for repeat in range(int(cfg["qa"].get("num_repeats", 1))):
                output = run_qa(
                    qubo, trial.embedding,
                    solver_id=target.solver_id or "unknown",
                    graph_id=target.graph_id, cfg=cfg,
                    embedding_seed=trial.embedding_seed, qa_repeat_id=repeat,
                )
                qpu_calls += 1
                qpu_runtime_total += float(output.get("qpu_runtime", 0.0) or 0.0)

                # 이 embedding 의 sample 만으로 독립 평가한다.
                selected, decoded, select_status = select_feasible_sample(
                    qubo, instance, cuts, output.get("samples", []), cfg
                )
                quality: dict[str, Any] = {
                    "select_status": str(select_status),
                    "num_decoded_samples": len(decoded),
                    "best_energy": decoded[0].energy if decoded else None,
                    "mp_feasible_sample_rate": feasible_sample_rate(decoded),
                }
                if selected is not None:
                    quality.update(
                        {f"selected_{k}": v for k, v in selected.as_record().items()}
                    )

                self.qa_records.append(
                    {
                        "record_type": "qa_run_per_embedding",
                        "instance_id": self.instance_id,
                        "benders_iteration": iteration,
                        "num_benders_cuts": len(cuts),
                        "topology_label": target.label,
                        "solver_id": target.solver_id,
                        "graph_id": target.graph_id,
                        "target_graph_fingerprint": target.fingerprint,
                        "solver_snapshot_timestamp": target.snapshot_timestamp,
                        "source_graph_fingerprint": fingerprint,
                        "embedding_seed": trial.embedding_seed,
                        "qa_repeat_id": repeat,
                        "embedding_file": embedding_files.get(int(trial.embedding_seed)),
                        "embedding_physical_qubits": trial.fields.get("physical_qubits"),
                        "embedding_max_chain_length": trial.fields.get("max_chain_length"),
                        "used_for_trajectory": chosen is None,
                        **quality,
                        **{k: v for k, v in output.items() if k != "samples"},
                    }
                )

                if self.verbose:
                    logger.info(
                        "[%s / %s] iter %d seed=%s repeat=%d: %s, best energy=%s, "
                        "feasible rate=%.3f, qpu %.2fs",
                        self.instance_id, target.label, iteration,
                        trial.embedding_seed, repeat, output.get("status"),
                        f"{quality['best_energy']:.6g}" if quality["best_energy"] is not None else "None",
                        quality["mp_feasible_sample_rate"],
                        float(output.get("qpu_runtime", 0.0) or 0.0),
                    )

                if chosen is None:
                    chosen = (trial, output)

                # 할당 시간이 소진되면 남은 seed 를 더 제출해도 전부 실패한다.
                # 즉시 중단하고 상태를 그대로 올려보낸다.
                if str(output.get("status")) == str(Status.QPU_QUOTA_EXHAUSTED):
                    if self.verbose:
                        logger.error(
                            "[%s / %s] iter %d: QPU 할당 시간 소진. 남은 seed 제출을 중단한다",
                            self.instance_id, target.label, iteration,
                        )
                    return {
                        "samples": output.get("samples", []),
                        "status": str(Status.QPU_QUOTA_EXHAUSTED),
                        "message": output.get("message", ""),
                        "trajectory_embedding_seed": trial.embedding_seed,
                        "qpu_runtime": qpu_runtime_total,
                        "num_qpu_calls": qpu_calls,
                        **base,
                    }

        trial, output = chosen
        return {
            "samples": output.get("samples", []),
            "status": output.get("status", str(Status.SUCCESS)),
            "trajectory_embedding_seed": trial.embedding_seed,
            "trajectory_qa_repeat_id": 0,
            "trajectory_embedding_file": embedding_files.get(int(trial.embedding_seed)),
            "selection_rule": "first_successful_embedding_seed_in_config_order",
            "qa_experiment_design": "A_independent_evaluation_fixed_trajectory",
            "mean_chain_break_fraction": output.get("mean_chain_break_fraction"),
            "max_chain_break_fraction": output.get("max_chain_break_fraction"),
            "chain_strength": output.get("chain_strength"),
            "qpu_runtime": qpu_runtime_total,
            "num_qpu_calls": qpu_calls,
            **base,
        }
