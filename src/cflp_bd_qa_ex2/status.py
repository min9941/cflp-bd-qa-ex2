"""실행 상태 enum 정의.

본 프로젝트는 실패를 삭제하거나 성공으로 바꾸지 않는다. 모든 실패는
아래 enum 중 하나로 기록되어 결과 파일에 그대로 남는다.

상태는 세 계층으로 분리한다.

- ``Status``          : 범용 실행 상태 (명세 22절)
- ``DualMinimalityStatus`` : dual guard 진단 결과 (진단 전용)
- ``DualGuardAction``      : dual guard 발동 시 취한 조치

``DualMinimalityStatus``와 ``DualGuardAction``은 진단 field이며
Benders run의 ``termination_status``를 덮어쓰지 않는다.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """문자열로 직렬화되는 enum 기반 클래스."""

    def __str__(self) -> str:  # pragma: no cover - 표현용
        return str(self.value)


class Status(StrEnum):
    """실행 결과 상태.

    명세 22절의 필수 상태와, 제한 라이선스/정확 검증을 위해 추가한
    상태를 함께 포함한다.
    """

    # --- 성공 계열 ---
    SUCCESS = "SUCCESS"
    OPTIMAL = "OPTIMAL"

    # --- 정상 종료이지만 최적이 아닌 경우 ---
    TIME_LIMIT = "TIME_LIMIT"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    STALLED_DUPLICATE_CUT = "STALLED_DUPLICATE_CUT"

    # --- 샘플/penalty 관련 ---
    NO_FEASIBLE_SAMPLE = "NO_FEASIBLE_SAMPLE"
    PENALTY_INSUFFICIENT = "PENALTY_INSUFFICIENT"

    # --- embedding 관련 ---
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    EMBEDDING_INVALID = "EMBEDDING_INVALID"

    # --- QPU 접근 관련 ---
    NO_QPU_ACCESS = "NO_QPU_ACCESS"
    # 계정/프로젝트의 solver access time 이 소진된 경우.
    # 접근 권한이 없는 것(NO_QPU_ACCESS)과도, 제출 실패(SOLVER_FAILED)와도 다르다.
    # 재시도해도 소용없으므로 trajectory 를 중단한다.
    QPU_QUOTA_EXHAUSTED = "QPU_QUOTA_EXHAUSTED"
    AMBIGUOUS_QPU_SELECTION = "AMBIGUOUS_QPU_SELECTION"
    TOPOLOGY_MISMATCH = "TOPOLOGY_MISMATCH"

    # --- encoding 관련 ---
    CUT_SLACK_RANGE_INVALID = "CUT_SLACK_RANGE_INVALID"
    THETA_RANGE_INVALID = "THETA_RANGE_INVALID"
    ENCODING_INVALID = "ENCODING_INVALID"

    # --- 모델 상태 ---
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    # Gurobi 의 INF_OR_UNBD. dual reductions 를 끄고 재최적화하기 전에는
    # infeasible 과 unbounded 를 구분할 수 없으므로 별도 상태로 둔다.
    INFEASIBLE_OR_UNBOUNDED = "INFEASIBLE_OR_UNBOUNDED"
    SOLVER_FAILED = "SOLVER_FAILED"
    ERROR = "ERROR"

    # --- Gurobi 라이선스 관련 (제한 라이선스 환경 대응) ---
    GUROBI_LICENSE_UNAVAILABLE = "GUROBI_LICENSE_UNAVAILABLE"
    GUROBI_SIZE_LIMIT = "GUROBI_SIZE_LIMIT"

    # --- 정확 검증(exhaustive / MIQP) 관련 ---
    EXACT_QUBO_VALIDATED = "EXACT_QUBO_VALIDATED"
    EXACT_QUBO_SKIPPED_TOO_LARGE = "EXACT_QUBO_SKIPPED_TOO_LARGE"

    # --- dual guard 위반으로 trajectory를 중단한 경우 ---
    DUAL_NOT_MINIMAL = "DUAL_NOT_MINIMAL"


class DualMinimalityStatus(StrEnum):
    """Dual minimality guard의 진단 결과.

    이 값은 진단 전용이며 ``termination_status``를 절대 덮어쓰지 않는다.
    """

    MINIMAL = "MINIMAL"
    DUAL_NOT_MINIMAL = "DUAL_NOT_MINIMAL"
    NOT_CHECKED = "NOT_CHECKED"


class DualGuardAction(StrEnum):
    """Dual guard 위반이 감지되었을 때 실제로 취한 조치."""

    NONE = "NONE"
    CONTINUE_WITH_RAW_DUAL = "CONTINUE_WITH_RAW_DUAL"
    STOP_RUN = "STOP_RUN"


class DuplicateCutStatus(StrEnum):
    """Duplicate-cut 진단 결과.

    이 값은 진단 전용이며 ``termination_status`` 를 덮어쓰지 않는다.
    duplicate cut 이 관측되어도 gap 이 개선되고 있으면 run 은 계속되며,
    그 경우 termination_status 는 다른 값(예: ``OPTIMAL``, ``MAX_ITERATIONS``)이 된다.
    """

    NONE = "NONE"
    DUPLICATE_DETECTED = "DUPLICATE_DETECTED"
    NOT_CHECKED = "NOT_CHECKED"


class TrajectorySource(StrEnum):
    """Benders cut trajectory를 생성한 주체."""

    GUROBI_CONTROLLED = "gurobi_controlled"
    SA_END_TO_END = "sa_end_to_end"
    QA_END_TO_END = "qa_end_to_end"


class TargetMode(StrEnum):
    """Embedding target graph의 종류."""

    IDEAL = "ideal"
    ACTUAL = "actual"


NA = "NA"
"""실패한 trial에서 chain/physical-qubit metric을 채울 때 사용하는 값."""
