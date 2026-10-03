"""Master Problem 의 QUBO 변환.

    F_QUBO = F_obj
             + penalty_alpha_capacity * r_capacity^2
             + sum_k penalty_alpha_cut_k * r_k^2

여기서

    F_obj      = sum_j f_j y_j + theta(t)
    r_capacity = sum_j s_j y_j - D - q_cap
    r_k        = theta(t) + sum_j s_j v_j^k y_j - sum_i u_i^k - q_k

모든 변수는 binary 이며 ``x^2 = x`` 를 이용해 제곱을 전개한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..benders.cuts import OptimalityCut
from ..data.generator import Instance
from ..encoding.slack import (
    CapacitySlackEncoding,
    CutSlackEncoding,
    build_capacity_slack_encoding,
    build_cut_slack_encoding,
    summarize_cut_slack_bits,
)
from ..encoding.theta import ThetaEncoding, build_theta_encoding
from .penalty import PenaltySpec, build_penalty_spec

ROLE_Y = "y"
ROLE_THETA = "theta_bit"
ROLE_CAPACITY_SLACK = "capacity_slack_bit"
ROLE_CUT_SLACK = "cut_slack_bit"


class AffineForm:
    """binary 변수에 대한 affine 식 ``sum_v c_v x_v + const``."""

    def __init__(self) -> None:
        self.coefficients: dict[str, float] = {}
        self.constant: float = 0.0

    def add(self, variable: str, coefficient: float) -> "AffineForm":
        """변수 항을 더한다."""
        if coefficient != 0.0:
            self.coefficients[variable] = self.coefficients.get(variable, 0.0) + coefficient
        return self

    def add_constant(self, value: float) -> "AffineForm":
        """상수항을 더한다."""
        self.constant += value
        return self

    def evaluate(self, assignment: dict[str, int]) -> float:
        """주어진 0/1 배정에서 식의 값을 계산한다."""
        return self.constant + sum(
            coef * int(assignment[var]) for var, coef in self.coefficients.items()
        )

    def square_into(
        self,
        linear: dict[str, float],
        quadratic: dict[tuple[str, str], float],
        weight: float,
    ) -> float:
        """``weight * (self)^2`` 를 linear/quadratic 에 누적하고 offset 을 반환한다.

        binary 변수의 ``x^2 = x`` 성질을 이용한다.
        """
        items = list(self.coefficients.items())
        for var, coef in items:
            linear[var] = linear.get(var, 0.0) + weight * (coef * coef + 2.0 * self.constant * coef)
        for a in range(len(items)):
            var_a, coef_a = items[a]
            for b in range(a + 1, len(items)):
                var_b, coef_b = items[b]
                key = (var_a, var_b) if var_a <= var_b else (var_b, var_a)
                quadratic[key] = quadratic.get(key, 0.0) + weight * 2.0 * coef_a * coef_b
        return weight * self.constant * self.constant


@dataclass
class MasterQUBO:
    """생성된 MP QUBO 와 그 metadata."""

    linear: dict[str, float]
    quadratic: dict[tuple[str, str], float]
    offset: float
    variable_labels: list[str]
    variable_roles: dict[str, str]
    constraint_metadata: dict[str, Any]
    theta_encoding: ThetaEncoding
    capacity_slack: CapacitySlackEncoding
    cut_slacks: list[CutSlackEncoding]
    penalty: PenaltySpec
    residual_forms: dict[str, AffineForm] = field(default_factory=dict)
    objective_form: AffineForm | None = None

    @property
    def num_variables(self) -> int:
        """logical variable 수."""
        return len(self.variable_labels)

    def energy(self, assignment: dict[str, int]) -> float:
        """주어진 0/1 배정의 QUBO energy 를 직접 계산한다."""
        total = self.offset
        for var, coef in self.linear.items():
            total += coef * int(assignment[var])
        for (a, b), coef in self.quadratic.items():
            total += coef * int(assignment[a]) * int(assignment[b])
        return total

    def to_bqm(self):
        """``dimod.BinaryQuadraticModel`` 로 변환한다."""
        import dimod

        return dimod.BinaryQuadraticModel(
            dict(self.linear), dict(self.quadratic), float(self.offset), dimod.BINARY
        )

    def variables_by_role(self, role: str) -> list[str]:
        """특정 role 의 변수 목록을 순서대로 반환한다."""
        return [v for v in self.variable_labels if self.variable_roles[v] == role]


def build_master_qubo(
    inst: Instance,
    cuts: Sequence[OptimalityCut],
    cfg: dict,
) -> MasterQUBO:
    """현재 cut 집합에 대한 MP QUBO 를 만든다.

    Args:
        inst: 대상 인스턴스.
        cuts: 활성 optimality cut 목록.
        cfg: 전체 설정 dict.

    Returns:
        생성된 :class:`MasterQUBO`.

    Raises:
        CutSlackRangeInvalid: cut slack 범위가 허용 오차를 넘어 음수인 경우.
    """
    enc_cfg = cfg["encoding"]
    theta_encoding = build_theta_encoding(
        inst.objective_upper_bound, int(enc_cfg["theta"]["bits"])
    )
    capacity_slack = build_capacity_slack_encoding(
        inst.total_capacity, inst.total_demand, float(enc_cfg["capacity_slack"]["delta"])
    )

    cut_slacks: list[CutSlackEncoding] = []
    for cut in cuts:
        cut_slacks.append(
            build_cut_slack_encoding(
                cut_index=cut.index,
                theta_upper_bound=theta_encoding.theta_upper_bound,
                capacity=inst.capacity.astype(float),
                dual_v=cut.v,
                dual_u_sum=cut.rhs_constant,
                theta_delta=theta_encoding.theta_delta,
                delta_ratio_to_theta=float(enc_cfg["cut_slack"]["delta_ratio_to_theta"]),
                negative_tolerance=float(enc_cfg["cut_slack"].get("negative_tolerance", 1e-6)),
            )
        )

    penalty = build_penalty_spec(inst.objective_upper_bound, len(cuts), cfg)

    # --- 변수 목록과 role ---
    labels: list[str] = []
    roles: dict[str, str] = {}

    y_names = [f"y_{j}" for j in range(inst.n_facilities)]
    for name in y_names:
        labels.append(name)
        roles[name] = ROLE_Y

    theta_names = theta_encoding.encoding.variable_names()
    for name in theta_names:
        labels.append(name)
        roles[name] = ROLE_THETA

    cap_names = capacity_slack.encoding.variable_names()
    for name in cap_names:
        labels.append(name)
        roles[name] = ROLE_CAPACITY_SLACK

    for slack in cut_slacks:
        for name in slack.encoding.variable_names():
            labels.append(name)
            roles[name] = ROLE_CUT_SLACK

    # --- theta affine form ---
    theta_form = AffineForm()
    for name, coef in zip(theta_names, theta_encoding.encoding.coefficients()):
        theta_form.add(name, coef)

    # --- objective ---
    objective = AffineForm()
    for j, name in enumerate(y_names):
        objective.add(name, float(inst.fixed_cost[j]))
    for name, coef in theta_form.coefficients.items():
        objective.add(name, coef)

    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    offset = 0.0

    for var, coef in objective.coefficients.items():
        linear[var] = linear.get(var, 0.0) + coef
    offset += objective.constant

    # --- capacity equality residual ---
    capacity_residual = AffineForm()
    for j, name in enumerate(y_names):
        capacity_residual.add(name, float(inst.capacity[j]))
    capacity_residual.add_constant(-float(inst.total_demand))
    for name, coef in zip(cap_names, capacity_slack.encoding.coefficients()):
        capacity_residual.add(name, -coef)
    offset += capacity_residual.square_into(linear, quadratic, penalty.penalty_alpha_capacity)

    residual_forms = {"capacity": capacity_residual}

    # --- cut equality residuals ---
    for cut, slack, alpha in zip(cuts, cut_slacks, penalty.penalty_alpha_cuts):
        residual = AffineForm()
        for name, coef in theta_form.coefficients.items():
            residual.add(name, coef)
        for j, name in enumerate(y_names):
            residual.add(name, float(cut.y_coefficients[j]))
        residual.add_constant(-float(cut.rhs_constant))
        for name, coef in zip(slack.encoding.variable_names(), slack.encoding.coefficients()):
            residual.add(name, -coef)
        offset += residual.square_into(linear, quadratic, alpha)
        residual_forms[f"cut_{cut.index}"] = residual

    # 0 계수 항 제거 (수치적으로 정확히 0인 항)
    linear = {k: v for k, v in linear.items() if v != 0.0}
    quadratic = {k: v for k, v in quadratic.items() if v != 0.0}

    constraint_metadata = {
        "num_cuts": len(cuts),
        "cut_indices": [c.index for c in cuts],
        "capacity_slack_bits": capacity_slack.capacity_slack_bits,
        "capacity_slack_max": capacity_slack.capacity_slack_max,
        **summarize_cut_slack_bits(cut_slacks),
        **theta_encoding.as_record(),
        **penalty.as_record(),
    }

    return MasterQUBO(
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        variable_labels=labels,
        variable_roles=roles,
        constraint_metadata=constraint_metadata,
        theta_encoding=theta_encoding,
        capacity_slack=capacity_slack,
        cut_slacks=cut_slacks,
        penalty=penalty,
        residual_forms=residual_forms,
        objective_form=objective,
    )
