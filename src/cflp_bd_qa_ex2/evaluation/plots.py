"""Figure 생성 (notebook 06 용).

모든 figure 의 text 는 **영어**로 작성한다 (한글 렌더링 tofu 방지).
성공률, 자원 사용량, solution quality, MP feasibility 를 하나의 "우승"
metric 으로 합치지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGSIZE = (7.0, 4.2)


def _finish(fig: plt.Figure, out_dir: str | Path, name: str) -> Path:
    """Figure 를 저장하고 닫는다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """``NA`` 문자열이 섞인 열을 숫자로 변환한다."""
    return pd.to_numeric(frame[column], errors="coerce")


def plot_final_gap_by_solver(frame: pd.DataFrame, out_dir: str | Path) -> Path:
    """1. Final objective gap by solver."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = frame.dropna(subset=["final_gap_percent"])
    if data.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        # instance 를 섞지 않는다. x 축은 instance_id 로 둔다.
        data = data.sort_values(["instance_name", "instance_id"])
        for source, group in data.groupby("trajectory_source"):
            ax.scatter(group["instance_id"], group["final_gap_percent"], label=str(source), s=45)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.legend(title="Trajectory source")
    ax.set_xlabel("Instance size")
    ax.set_ylabel("Final objective gap [%]")
    ax.set_title("Final objective gap by solver")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "fig01_final_gap_by_solver")


def plot_mp_feasible_rate(frame: pd.DataFrame, out_dir: str | Path) -> Path:
    """2. MP feasible-sample rate by solver."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = frame.dropna(subset=["mp_feasible_sample_rate"])
    if data.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        # 서로 다른 instance 를 한 선으로 잇지 않는다.
        keys = [k for k in ("instance_id", "trajectory_source", "configuration_hash")
                if k in data.columns]
        for key, group in data.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            ax.plot(
                group.sort_values("benders_iteration")["benders_iteration"],
                group.sort_values("benders_iteration")["mp_feasible_sample_rate"],
                marker="o",
                markersize=3.5,
                linewidth=1.0,
                label=" / ".join(str(part)[:12] for part in key[:2]),
            )
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 10:
            ax.legend(fontsize=6, ncol=2)
        else:
            ax.text(0.99, 0.02, f"{len(labels)} series", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, alpha=0.7)
    ax.set_xlabel("Benders iteration")
    ax.set_ylabel("MP-feasible sample rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("MP feasible-sample rate")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "fig02_mp_feasible_rate")


def plot_bound_trajectory(
    frame: pd.DataFrame, out_dir: str | Path, instance_id: str, suffix: bool = True
) -> Path:
    """3. Benders LB/UB trajectory.

    Args:
        frame: iteration 기록.
        out_dir: 출력 디렉터리.
        instance_id: 대상 인스턴스.
        suffix: True 면 파일명에 instance_id 를 붙여 **덮어쓰기를 막는다.**
            여러 인스턴스를 반복해 그릴 때 반드시 True 여야 한다.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = frame[frame["instance_id"] == instance_id]
    if data.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        for source, group in data.groupby("trajectory_source"):
            group = group.sort_values("benders_iteration")
            ax.plot(
                group["benders_iteration"],
                group["certified_lower_bound"],
                marker="o",
                label=f"{source} LB",
            )
            ax.plot(
                group["benders_iteration"],
                group["best_upper_bound"],
                marker="s",
                linestyle="--",
                label=f"{source} UB",
            )
        ax.legend(fontsize=8)
    ax.set_xlabel("Benders iteration")
    ax.set_ylabel("Objective")
    ax.set_title(f"Benders LB/UB trajectory ({instance_id})")
    ax.grid(alpha=0.3)
    name = f"fig03_bound_trajectory_{instance_id}" if suffix else "fig03_bound_trajectory"
    return _finish(fig, out_dir, name)


def plot_iteration_series(
    frame: pd.DataFrame,
    out_dir: str | Path,
    column: str,
    ylabel: str,
    title: str,
    name: str,
    logy: bool = False,
) -> Path:
    """Iteration 대비 단일 metric 을 인스턴스별로 그린다."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if column not in frame.columns or frame.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        values = _numeric(frame, column)
        data = frame.assign(_value=values).dropna(subset=["_value"])

        # 서로 다른 instance seed / topology / embedding seed 를 한 선으로 잇지 않는다.
        # 존재하는 열만 골라 그룹 key 를 만든다.
        candidate_keys = [
            "instance_id",
            "trajectory_source",
            "topology_label",
            "embedding_seed",
        ]
        group_keys = [k for k in candidate_keys if k in data.columns]
        for key, group in data.groupby(group_keys):
            key = key if isinstance(key, tuple) else (key,)
            label = " / ".join(str(part) for part in key)
            ax.plot(
                group.sort_values("benders_iteration")["benders_iteration"],
                group.sort_values("benders_iteration")["_value"],
                marker="o",
                linewidth=1.0,
                markersize=3.5,
                alpha=0.8,
                label=label,
            )
        if logy:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        # 선이 너무 많으면 범례가 그림을 가린다. 개수만 표기한다.
        if len(labels) <= 10:
            ax.legend(fontsize=6, ncol=2)
        else:
            ax.text(
                0.99, 0.02, f"{len(labels)} series", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, alpha=0.7,
            )
    ax.set_xlabel("Benders iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, name)


def embedding_success_table(frame: pd.DataFrame, average: str = "micro") -> pd.DataFrame:
    """Embedding 성공률 집계표를 만든다 (plot 과 분리해 검증 가능하게 둔다).

    Args:
        frame: embedding trial 기록. ``instance_name``, ``topology_label``,
            ``instance_id``, ``success`` 열이 필요하다.
        average: ``"micro"`` 는 전체 trial 을 동일 가중으로 평균한다
            (iteration 이 많은 instance seed 가 더 큰 가중치를 받는다).
            ``"macro"`` 는 instance seed 별 성공률을 먼저 구한 뒤 동일 가중 평균한다.

    Returns:
        행이 ``instance_name``, 열이 ``topology_label`` 인 성공률 표.

    Raises:
        ValueError: ``average`` 가 ``"micro"`` 또는 ``"macro"`` 가 아닌 경우.
    """
    if average not in ("micro", "macro"):
        raise ValueError(f"average 는 'micro' 또는 'macro' 여야 한다: {average!r}")
    if frame.empty:
        return pd.DataFrame()

    if average == "macro":
        per_instance = (
            frame.groupby(["instance_name", "topology_label", "instance_id"])["success"]
            .mean()
            .reset_index()
        )
        return per_instance.pivot_table(
            index="instance_name", columns="topology_label", values="success", aggfunc="mean"
        )
    return frame.pivot_table(
        index="instance_name", columns="topology_label", values="success", aggfunc="mean"
    )


def plot_embedding_success_heatmap(
    frame: pd.DataFrame, out_dir: str | Path, average: str = "micro"
) -> Path:
    """11. Instance size x topology embedding success-rate heatmap.

    **세 축(instance seed, Benders iteration, embedding seed)이 평균된다.**
    어떤 평균인지 반드시 구분해야 한다.

    Args:
        frame: embedding trial 기록.
        out_dir: 출력 디렉터리.
        average: ``"micro"`` 는 전체 trial 을 동일 가중으로 평균한다
            (iteration 이 많은 instance seed 가 더 큰 가중치를 받는다).
            ``"macro"`` 는 instance seed 별 성공률을 먼저 구한 뒤 동일 가중 평균한다.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if frame.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _finish(fig, out_dir, "fig11_embedding_success_heatmap")

    table = embedding_success_table(frame, average=average)
    image = ax.imshow(table.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            value = table.values[r, c]
            if not np.isnan(value):
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="w", fontsize=8)
    label = (
        "Trial-weighted (micro) success rate"
        if average == "micro"
        else "Instance-weighted (macro) success rate"
    )
    fig.colorbar(image, ax=ax, label=label)
    ax.set_title(f"Embedding success rate ({average}): instance size x topology")
    return _finish(fig, out_dir, f"fig11_embedding_success_heatmap_{average}")


def plot_embedding_success_rate_by_iteration(
    frame: pd.DataFrame, out_dir: str | Path
) -> Path:
    """11b. Iteration 별 embedding 성공률 (평균으로 숨기지 않고 따로 본다)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if frame.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        grouped = frame.groupby(["instance_name", "topology_label", "benders_iteration"])[
            "success"
        ].mean().reset_index()
        for key, group in grouped.groupby(["instance_name", "topology_label"]):
            ax.plot(
                group["benders_iteration"], group["success"],
                marker="o", markersize=4, label=f"{key[0]} / {key[1]}",
            )
        ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Benders iteration")
    ax.set_ylabel("Embedding success rate")
    ax.set_title("Embedding success rate per iteration")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "fig11b_embedding_success_by_iteration")


def plot_ideal_vs_actual(frame: pd.DataFrame, out_dir: str | Path) -> Path:
    """12. Ideal graph vs actual working graph."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = frame.assign(physical=_numeric(frame, "physical_qubits")).dropna(subset=["physical"])
    if data.empty:
        ax.text(0.5, 0.5, "no actual-QPU data", ha="center", va="center")
    else:
        for mode, group in data.groupby("target_mode"):
            ax.scatter(
                pd.to_numeric(group["logical_variables"], errors="coerce"),
                group["physical"],
                label=str(mode),
                s=40,
                alpha=0.75,
            )
        ax.legend(title="Target mode")
    ax.set_xlabel("Logical variables")
    ax.set_ylabel("Physical qubits")
    ax.set_title("Ideal target vs actual working graph")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "fig12_ideal_vs_actual")


def plot_chain_break_vs_feasibility(frame: pd.DataFrame, out_dir: str | Path) -> Path:
    """13. Chain-break fraction vs MP feasibility."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    needed = {"mean_chain_break_fraction", "mp_feasible_sample_rate"}
    data = frame.dropna(subset=list(needed)) if needed <= set(frame.columns) else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "no QA data (QPU not accessed)", ha="center", va="center")
    else:
        ax.scatter(data["mean_chain_break_fraction"], data["mp_feasible_sample_rate"], s=40)
    ax.set_xlabel("Mean chain-break fraction")
    ax.set_ylabel("MP-feasible sample rate")
    ax.set_title("Chain-break fraction vs MP feasibility")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "fig13_chain_break_vs_feasibility")


def plot_slack_bit_composition(frame: pd.DataFrame, out_dir: str | Path) -> Path:
    """14. Theta / cut / capacity slack bit composition over iterations."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    columns = ["theta_bits", "capacity_slack_bits", "cut_slack_bits_total"]
    if frame.empty or not set(columns) <= set(frame.columns):
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _finish(fig, out_dir, "fig14_slack_bit_composition")

    grouped = frame.groupby("benders_iteration")[columns].mean()
    ax.stackplot(
        grouped.index,
        grouped["theta_bits"],
        grouped["capacity_slack_bits"],
        grouped["cut_slack_bits_total"],
        labels=["theta bits", "capacity slack bits", "cut slack bits"],
        alpha=0.85,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Benders iteration")
    ax.set_ylabel("Number of bits")
    ax.set_title("Bit composition over Benders iterations")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir, "fig14_slack_bit_composition")


def generate_all_figures(
    iteration_frame: pd.DataFrame,
    run_frame: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    out_dir: str | Path,
) -> list[Path]:
    """명세 24절이 요구하는 figure 를 모두 생성한다.

    Returns:
        생성된 figure 경로 목록.
    """
    paths: list[Path] = []
    paths.append(plot_final_gap_by_solver(run_frame, out_dir))
    paths.append(plot_mp_feasible_rate(iteration_frame, out_dir))

    instance_id = (
        iteration_frame["instance_id"].iloc[0] if not iteration_frame.empty else "none"
    )
    # 대표 인스턴스 하나를 고정 이름으로 남긴다.
    # 인스턴스별 figure 가 필요하면 plot_bound_trajectory(..., suffix=True) 를 쓴다.
    paths.append(plot_bound_trajectory(iteration_frame, out_dir, instance_id, suffix=False))

    series: Sequence[tuple[str, str, str, str, bool]] = (
        ("logical_variables", "Logical variables", "Logical variables per iteration", "fig04_logical_variables", False),
        ("logical_edges", "Logical edges", "Logical edges per iteration", "fig05_logical_edges", False),
        ("qubo_density", "QUBO density", "QUBO density per iteration", "fig06_qubo_density", False),
        ("qubo_coefficient_range", "Coefficient range (max/min)", "QUBO coefficient range per iteration", "fig07_coefficient_range", True),
    )
    for column, ylabel, title, name, logy in series:
        paths.append(plot_iteration_series(iteration_frame, out_dir, column, ylabel, title, name, logy))

    embed_series: Sequence[tuple[str, str, str, str, bool]] = (
        ("physical_qubits", "Physical qubits", "Physical qubits per iteration", "fig08_physical_qubits", False),
        ("mean_chain_length", "Mean chain length", "Mean chain length per iteration", "fig09a_mean_chain_length", False),
        ("max_chain_length", "Max chain length", "Max chain length per iteration", "fig09b_max_chain_length", False),
        ("embedding_time", "Embedding time [s]", "Embedding time per iteration", "fig10_embedding_time", False),
    )
    for column, ylabel, title, name, logy in embed_series:
        paths.append(
            plot_iteration_series(embedding_frame, out_dir, column, ylabel, title, name, logy)
        )

    # micro 와 macro 를 모두 남긴다. 하나로 합치지 않는다.
    paths.append(plot_embedding_success_heatmap(embedding_frame, out_dir, average="micro"))
    paths.append(plot_embedding_success_heatmap(embedding_frame, out_dir, average="macro"))
    paths.append(plot_embedding_success_rate_by_iteration(embedding_frame, out_dir))
    paths.append(plot_ideal_vs_actual(embedding_frame, out_dir))
    paths.append(plot_chain_break_vs_feasibility(iteration_frame, out_dir))
    paths.append(plot_slack_bit_composition(iteration_frame, out_dir))
    return paths
