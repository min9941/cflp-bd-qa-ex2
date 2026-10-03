"""Benders decomposition 루프.

세 종류의 trajectory 를 하나의 루프로 지원한다.

- ``gurobi_controlled``: MP 후보를 Gurobi continuous MP 에서 얻는다.
- ``sa_end_to_end``    : MP 후보를 SA 가 푼 QUBO sample 에서 얻는다.
- ``qa_end_to_end``    : MP 후보를 QA 가 푼 QUBO sample 에서 얻는다.

**어느 경우에도** 동일 iteration 의 continuous MP 를 Gurobi 로 별도 해결하여
certified lower bound 를 얻는다. QA/SA 목적값은 lower bound 로 쓰지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from ..data.generator import Instance
from ..models.cflp import evaluate_original_objective
from ..models.master import mp_violations, solve_continuous_mp, solve_encoded_mp
from ..models.subproblem import (
    DualSolution,
    check_dual_minimality,
    dual_continuation_allowed,
    solve_subproblem,
)
from ..qubo.builder import build_master_qubo
from ..qubo.decode import DecodedSample, feasible_sample_rate, select_feasible_sample
from ..qubo.diagnostics import logical_graph, qubo_metrics, role_counts, source_graph_fingerprint
from ..status import (
    DualGuardAction,
    DualMinimalityStatus,
    DuplicateCutStatus,
    Status,
    TrajectorySource,
)
from .certificate import BoundState
from .cuts import OptimalityCut, build_optimality_cut
from .trace import BendersTrace, IterationRecord, RunRecord


@dataclass
class CandidateResult:
    """한 iteration 에서 MP 후보를 얻은 결과."""

    status: Status
    y: np.ndarray | None = None
    theta: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)


class MasterProvider(Protocol):
    """MP 후보 생성기 인터페이스."""

    source: TrajectorySource

    def candidate(
        self,
        inst: Instance,
        cuts: Sequence[OptimalityCut],
        iteration: int,
        cfg: dict,
        continuous_payload: dict[str, Any],
    ) -> CandidateResult:
        """MP 후보 ``(y, theta)`` 를 만든다."""
        ...


class GurobiMasterProvider:
    """Gurobi continuous MP 의 해를 그대로 후보로 사용한다."""

    source = TrajectorySource.GUROBI_CONTROLLED

    def candidate(
        self,
        inst: Instance,
        cuts: Sequence[OptimalityCut],
        iteration: int,
        cfg: dict,
        continuous_payload: dict[str, Any],
    ) -> CandidateResult:
        """Continuous MP 결과에서 후보를 추출한다."""
        if "y" not in continuous_payload:
            return CandidateResult(status=Status.SOLVER_FAILED)
        return CandidateResult(
            status=Status.SUCCESS,
            y=np.asarray(continuous_payload["y"], dtype=int),
            theta=float(continuous_payload["theta"]),
        )


class QuboMasterProvider:
    """QUBO 를 만들어 sampler 로 후보를 얻는다 (SA 또는 QA)."""

    def __init__(
        self,
        source: TrajectorySource,
        sampler: Callable[..., dict[str, Any]],
        collect_qubo_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            source: trajectory 종류.
            sampler: ``(MasterQUBO, cfg, context=...) -> dict`` 형태의 샘플러.
                ``context`` 는 ``instance``/``cuts``/``iteration`` 을 담으며,
                샘플러가 embedding 별로 직접 decode·평가할 때 사용한다.
                반환 dict 는 ``samples`` key 를 가져야 하고,
                실패 시 ``status`` 로 실패 종류를 알려야 한다.
            collect_qubo_records: QUBO metric 을 모아 둘 리스트(선택).
        """
        self.source = source
        self.sampler = sampler
        self.collect_qubo_records = collect_qubo_records

    def candidate(
        self,
        inst: Instance,
        cuts: Sequence[OptimalityCut],
        iteration: int,
        cfg: dict,
        continuous_payload: dict[str, Any],
    ) -> CandidateResult:
        """QUBO 를 만들고 샘플링하여 MP-feasible 한 첫 sample 을 고른다."""
        build_start = time.perf_counter()
        qubo = build_master_qubo(inst, cuts, cfg)
        build_time = time.perf_counter() - build_start

        graph = logical_graph(qubo)
        metrics = qubo_metrics(qubo)
        metrics.update(role_counts(qubo))
        metrics.update(qubo.constraint_metadata)
        metrics["source_graph_fingerprint"] = source_graph_fingerprint(graph)
        metrics["qubo_build_time"] = build_time

        sampler_out = self.sampler(
            qubo, cfg, context={"instance": inst, "cuts": list(cuts), "iteration": iteration}
        )

        fields: dict[str, Any] = dict(metrics)
        fields.update({k: v for k, v in sampler_out.items() if k != "samples"})

        # sampler 자체가 실패했으면 그 status 를 그대로 전파한다.
        # 빈 sample 목록을 NO_FEASIBLE_SAMPLE 로 바꾸면
        # "QPU 미접근"과 "feasible sample 없음"이 구분되지 않는다.
        sampler_status = sampler_out.get("status")
        if sampler_status is not None and str(sampler_status) != str(Status.SUCCESS):
            fields["sampler_status"] = str(sampler_status)
            return CandidateResult(status=Status(str(sampler_status)), fields=fields)

        selected, decoded_all, status = select_feasible_sample(
            qubo, inst, cuts, sampler_out["samples"], cfg
        )
        fields["mp_feasible_sample_rate"] = feasible_sample_rate(decoded_all)
        fields["num_decoded_samples"] = len(decoded_all)
        if decoded_all:
            fields["best_energy"] = decoded_all[0].energy
        if self.collect_qubo_records is not None:
            self.collect_qubo_records.append(dict(fields))

        if selected is None:
            return CandidateResult(status=Status.NO_FEASIBLE_SAMPLE, fields=fields)

        fields.update(selected.as_record())
        return CandidateResult(
            status=Status.SUCCESS,
            y=selected.decoded_y,
            theta=selected.decoded_theta,
            fields=fields,
        )


def _guard_action(cfg: dict, source: TrajectorySource) -> str:
    """trajectory 별 dual guard 조치를 읽는다."""
    actions = cfg["benders"]["dual_minimality_guard"]["action_by_trajectory"]
    return actions.get(str(source), "stop_run")


def run_benders(
    inst: Instance,
    provider: MasterProvider,
    cfg: dict,
    solve_encoded_each_iteration: bool = False,
) -> BendersTrace:
    """하나의 인스턴스에 대해 Benders 루프를 실행한다.

    Args:
        inst: 대상 인스턴스.
        provider: MP 후보 생성기.
        cfg: 전체 설정 dict.
        solve_encoded_each_iteration: True 면 매 iteration 마다 encoded MP 도
            Gurobi 로 풀어 discretization 영향을 측정한다.

    Returns:
        :class:`BendersTrace`.
    """
    bcfg = cfg["benders"]
    guard_cfg = bcfg["dual_minimality_guard"]
    tolerance = float(bcfg["relative_gap_tolerance"])
    max_iterations = int(bcfg["max_iterations"])
    audit_tol = float(cfg["gurobi"]["numerical_audit_tolerance"])

    trace = BendersTrace()
    cuts: list[OptimalityCut] = []
    bounds = BoundState()
    seen_signatures: set[str] = set()
    seen_y_signatures: set[str] = set()
    # 정체 판정은 **(y, cut) 쌍** 기준이다. 두 집합을 따로 보면
    # 과거에 (A, X) 와 (B, Y) 가 나온 뒤 처음 등장한 (A, Y) 를 반복으로 오판한다.
    seen_pairs: set[tuple[str, str]] = set()
    source = str(provider.source)

    totals = {
        "embedding_runtime": 0.0,
        "qpu_runtime": 0.0,
        "num_embedding_calls": 0,
        "num_qpu_calls": 0,
        "mp_runtime": 0.0,
        "continuous_mp_certificate_runtime": 0.0,
        "encoded_mp_runtime": 0.0,
        "qubo_runtime": 0.0,
        "sp_runtime": 0.0,
        "num_mp_calls": 0,
        "num_sp_calls": 0,
    }
    termination = Status.MAX_ITERATIONS
    termination_detail = ""
    dual_guard_action = DualGuardAction.NONE
    dual_status_last = DualMinimalityStatus.NOT_CHECKED

    # duplicate-cut 진단은 termination_status 와 **별개 field** 로 관리한다.
    duplicate_cut_status = DuplicateCutStatus.NONE
    num_duplicate_cuts = 0
    first_duplicate_iteration: int | None = None
    previous_gap = float("inf")

    run_start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        # --- 1. certified lower bound: 항상 Gurobi continuous MP ---
        continuous = solve_continuous_mp(inst, cuts, cfg)
        totals["continuous_mp_certificate_runtime"] += continuous.runtime
        totals["num_mp_calls"] += 1
        if continuous.status is not Status.OPTIMAL:
            termination = continuous.status
            termination_detail = f"continuous MP 실패: {continuous.message}"
            trace.add_iteration(
                IterationRecord(
                    instance_id=inst.instance_id,
                    instance_name=inst.name,
                    instance_seed=inst.seed,
                    trajectory_source=source,
                    benders_iteration=iteration,
                    num_benders_cuts=len(cuts),
                    fields={
                        "continuous_mp_status": str(continuous.status),
                        "message": continuous.message,
                    },
                )
            )
            break
        bounds.update_lower(continuous.objective, "continuous_mp_optimal", continuous.status)

        record_fields: dict[str, Any] = {
            "continuous_mp_status": str(continuous.status),
            "continuous_mp_objective": continuous.objective,
            "continuous_mp_bound": continuous.bound,
            "continuous_mp_runtime": continuous.runtime,
            "continuous_mp_y": continuous.payload.get("y"),
            "continuous_mp_theta": continuous.payload.get("theta"),
        }

        # --- 2. encoded MP (선택) ---
        if solve_encoded_each_iteration:
            from ..encoding.theta import build_theta_encoding

            theta_encoding = build_theta_encoding(
                inst.objective_upper_bound, int(cfg["encoding"]["theta"]["bits"])
            )
            encoded = solve_encoded_mp(inst, cuts, theta_encoding, cfg)
            totals["encoded_mp_runtime"] += encoded.runtime
            record_fields.update(
                {
                    "encoded_mp_status": str(encoded.status),
                    "encoded_mp_objective": encoded.objective,
                    "encoded_mp_runtime": encoded.runtime,
                    "encoded_mp_theta": encoded.payload.get("theta"),
                }
            )

        # --- 3. MP 후보 ---
        cand_start = time.perf_counter()
        candidate = provider.candidate(inst, cuts, iteration, cfg, continuous.payload)
        cand_time = time.perf_counter() - cand_start
        totals["mp_runtime"] += cand_time
        totals["qubo_runtime"] += float(candidate.fields.get("qubo_build_time", 0.0))
        for key, source_key in (
            ("embedding_runtime", "embedding_runtime"),
            ("qpu_runtime", "qpu_runtime"),
            ("num_embedding_calls", "num_embedding_trials"),
            ("num_qpu_calls", "num_qpu_calls"),
        ):
            totals[key] += candidate.fields.get(source_key, 0) or 0
        record_fields.update(candidate.fields)
        record_fields["candidate_status"] = str(candidate.status)
        record_fields["candidate_runtime"] = cand_time

        if candidate.status is not Status.SUCCESS or candidate.y is None:
            termination = candidate.status
            termination_detail = "MP 후보를 얻지 못했다"
            record_fields["certified_lower_bound"] = bounds.certified_lower_bound
            record_fields["best_upper_bound"] = bounds.best_upper_bound
            trace.add_iteration(
                IterationRecord(
                    instance_id=inst.instance_id,
                    instance_name=inst.name,
                    instance_seed=inst.seed,
                    trajectory_source=source,
                    benders_iteration=iteration,
                    num_benders_cuts=len(cuts),
                    fields=record_fields,
                )
            )
            break

        y = np.asarray(candidate.y, dtype=int)

        # --- 4. subproblem ---
        sp_outcome, dual = solve_subproblem(inst, y, cfg)
        totals["sp_runtime"] += sp_outcome.runtime
        totals["num_sp_calls"] += 1
        record_fields["sp_status"] = str(sp_outcome.status)
        record_fields["sp_runtime"] = sp_outcome.runtime

        if sp_outcome.status is not Status.OPTIMAL or dual is None:
            termination = sp_outcome.status
            termination_detail = (
                "SP 가 OPTIMAL 이 아니다. feasibility cut 을 임의 생성하지 않고 중단한다"
            )
            trace.add_iteration(
                IterationRecord(
                    instance_id=inst.instance_id,
                    instance_name=inst.name,
                    instance_seed=inst.seed,
                    trajectory_source=source,
                    benders_iteration=iteration,
                    num_benders_cuts=len(cuts),
                    fields=record_fields,
                )
            )
            break

        # --- 5. dual minimality guard ---
        if bool(guard_cfg.get("enabled", True)):
            dual = check_dual_minimality(
                dual, y, float(guard_cfg["atol"]), float(guard_cfg["rtol"])
            )
        dual_status_last = dual.minimality_status
        record_fields.update(dual.as_record())

        if dual.minimality_status is DualMinimalityStatus.DUAL_NOT_MINIMAL:
            action = _guard_action(cfg, provider.source)
            allowed, reason = dual_continuation_allowed(
                sp_outcome.status,
                dual,
                duality_tolerance=audit_tol,
                dual_feasibility_tolerance=float(cfg["gurobi"]["feasibility_tolerance"]),
            )
            record_fields["dual_guard_reason"] = reason
            if action == "stop_run" or not allowed:
                dual_guard_action = DualGuardAction.STOP_RUN
                record_fields["dual_guard_action"] = str(dual_guard_action)
                termination = Status.DUAL_NOT_MINIMAL
                termination_detail = (
                    f"dual minimality guard 위반으로 trajectory 중단 ({reason})"
                )
                trace.add_iteration(
                    IterationRecord(
                        instance_id=inst.instance_id,
                        instance_name=inst.name,
                        instance_seed=inst.seed,
                        trajectory_source=source,
                        benders_iteration=iteration,
                        num_benders_cuts=len(cuts),
                        fields=record_fields,
                    )
                )
                break
            dual_guard_action = DualGuardAction.CONTINUE_WITH_RAW_DUAL
            record_fields["dual_guard_action"] = str(dual_guard_action)
        else:
            record_fields["dual_guard_action"] = str(DualGuardAction.NONE)

        # --- 6. upper bound ---
        candidate_objective = evaluate_original_objective(inst, y, sp_outcome.objective)
        improved = bounds.update_upper(candidate_objective, y.tolist())
        record_fields.update(
            {
                "true_sp_value_for_decoded_y": sp_outcome.objective,
                "candidate_original_objective": candidate_objective,
                "upper_bound_improved": improved,
            }
        )

        violation = mp_violations(inst, y, float(candidate.theta or 0.0), cuts)
        record_fields.update(violation)
        record_fields["mp_feasible"] = violation["max_mp_violation"] <= float(
            cfg["validation"]["mp_feasibility_tolerance"]
        )

        # --- 7. bound 감사와 종료 판정 ---
        problem = bounds.check_bound_order(audit_tol)
        if problem is not None:
            termination = Status.ERROR
            termination_detail = problem
            record_fields["bound_audit"] = problem
            trace.add_iteration(
                IterationRecord(
                    instance_id=inst.instance_id,
                    instance_name=inst.name,
                    instance_seed=inst.seed,
                    trajectory_source=source,
                    benders_iteration=iteration,
                    num_benders_cuts=len(cuts),
                    fields=record_fields,
                )
            )
            break

        gap = bounds.relative_gap()
        record_fields.update(
            {
                "certified_lower_bound": bounds.certified_lower_bound,
                "best_upper_bound": bounds.best_upper_bound,
                "relative_gap": gap,
            }
        )

        # --- 8. cut 생성 ---
        cut = build_optimality_cut(
            index=len(cuts),
            capacity=inst.capacity.astype(float),
            u=dual.u,
            v=dual.v,
            iteration=iteration,
            dual_minimality_status=str(dual.minimality_status),
        )
        y_signature = ",".join(str(int(v)) for v in y)
        repeated_y = y_signature in seen_y_signatures
        duplicate = cut.signature in seen_signatures
        repeated_pair = (y_signature, cut.signature) in seen_pairs
        if duplicate:
            num_duplicate_cuts += 1
            duplicate_cut_status = DuplicateCutStatus.DUPLICATE_DETECTED
            if first_duplicate_iteration is None:
                first_duplicate_iteration = iteration

        # gap 이 실제로 줄고 있는지 별도로 판정한다.
        gap_improved = gap < previous_gap - tolerance
        record_fields["duplicate_cut"] = duplicate
        record_fields["repeated_y"] = repeated_y
        record_fields["repeated_y_cut_pair"] = repeated_pair
        record_fields["y_signature"] = y_signature
        record_fields["duplicate_cut_status"] = str(duplicate_cut_status)
        record_fields["num_duplicate_cuts"] = num_duplicate_cuts
        record_fields["gap_improved"] = gap_improved
        record_fields["previous_relative_gap"] = (
            previous_gap if previous_gap != float("inf") else None
        )
        record_fields["cut_signature"] = cut.signature
        # 이 cut 이 실제로 MP 에 추가되는지는 아래 종료 판정 이후에 결정된다.
        # 수렴하여 종료되면 이 cut 은 **한 번도 활성 cut 이 된 적이 없다.**
        # 따라서 embedding study 가 존재하지 않았던 MP 상태를 만들지 않도록
        # 활성화 여부를 명시적으로 기록한다.
        snapshot = {
            "instance_id": inst.instance_id,
            "trajectory_source": source,
            "benders_iteration": iteration,
            "cut_activated": False,
            **cut.as_record(),
        }
        trace.cut_snapshots.append(snapshot)

        trace.add_iteration(
            IterationRecord(
                instance_id=inst.instance_id,
                instance_name=inst.name,
                instance_seed=inst.seed,
                trajectory_source=source,
                benders_iteration=iteration,
                num_benders_cuts=len(cuts),
                fields=record_fields,
            )
        )

        if gap <= tolerance:
            termination = Status.OPTIMAL
            termination_detail = "relative gap tolerance 도달"
            break

        # 명세 7절: "**동일한 y** 와 동일한 cut 이 반복되고 **gap 이 줄지 않으면**".
        # (y, cut) 쌍이 그대로 반복되었을 때만 정체로 본다. 서로 다른 y 가 우연히
        # 같은 dual 을 주는 경우(distinct_y_same_cut)나, 같은 y 가 다른 cut 을 주는
        # 경우를 STALLED 로 오판하지 않는다.
        if repeated_pair and not gap_improved:
            termination = Status.STALLED_DUPLICATE_CUT
            termination_detail = (
                "동일한 cut 이 반복되고 gap 이 줄지 않는다. 임의 heuristic 을 적용하지 않는다"
            )
            break

        previous_gap = gap
        seen_y_signatures.add(y_signature)
        seen_pairs.add((y_signature, cut.signature))

        if duplicate:
            # 중복 cut 은 MP 에 아무 정보를 더하지 않으면서 slack bit 만 늘린다.
            # 따라서 추가하지 않는다. 이것은 해를 고치는 heuristic 이 아니라
            # 중복 제약을 넣지 않는 것일 뿐이며, 진단 field 에 그대로 기록된다.
            continue

        seen_signatures.add(cut.signature)
        cuts.append(cut)
        snapshot["cut_activated"] = True

    runtime = time.perf_counter() - run_start
    trace.run = RunRecord(
        instance_id=inst.instance_id,
        instance_name=inst.name,
        instance_seed=inst.seed,
        trajectory_source=source,
        fields={
            "benders_iterations": len(trace.iterations),
            "benders_runtime": runtime,
            **totals,
            "final_objective": bounds.best_upper_bound,
            "final_optimality_gap": bounds.relative_gap(),
            "certified_lower_bound": bounds.certified_lower_bound,
            "best_upper_bound": bounds.best_upper_bound,
            "best_y": bounds.best_y,
            "lower_bound_source": bounds.lower_bound_source,
            "termination_status": str(termination),
            "termination_detail": termination_detail,
            "dual_minimality_status": str(dual_status_last),
            "dual_guard_action": str(dual_guard_action),
            "duplicate_cut_status": str(duplicate_cut_status),
            "num_duplicate_cuts": num_duplicate_cuts,
            "first_duplicate_iteration": first_duplicate_iteration,
            "num_cuts_generated": len(cuts),
        },
    )
    return trace
