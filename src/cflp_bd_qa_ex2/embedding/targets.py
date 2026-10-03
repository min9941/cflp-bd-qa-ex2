"""Embedding target graph: 이상적 topology 와 실제 QPU working graph.

두 가지를 **절대 섞지 않는다**.

- **ideal target**: 결함이 없는 완전한 Pegasus P16 / Zephyr Z12 그래프.
  여기서는 QA sampling 을 수행하지 않는다.
- **actual working graph**: 실제 QPU 의 결함이 반영된 그래프.
  QA sampling 은 여기서만 수행한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..status import Status, TargetMode


@dataclass
class TargetGraph:
    """Embedding target 하나."""

    label: str
    topology: str
    topology_shape: int
    mode: TargetMode
    graph: nx.Graph
    solver_id: str | None = None
    chip_id: str | None = None
    graph_id: str = ""
    snapshot_timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_nodes(self) -> int:
        """target graph 의 노드 수."""
        return self.graph.number_of_nodes()

    @property
    def target_edges(self) -> int:
        """target graph 의 엣지 수."""
        return self.graph.number_of_edges()

    @property
    def fingerprint(self) -> str:
        """target graph 의 결정적 지문."""
        return target_graph_fingerprint(self.graph)

    def as_record(self) -> dict[str, Any]:
        """결과 저장용 dict (명세 14절)."""
        return {
            "topology": self.topology,
            "topology_label": self.label,
            "topology_size": self.topology_shape,
            "topology_shape": self.topology_shape,
            "target_mode": str(self.mode),
            "solver_id": self.solver_id,
            "chip_id": self.chip_id,
            "graph_id": self.graph_id,
            "target_nodes": self.target_nodes,
            "target_edges": self.target_edges,
            "target_graph_fingerprint": self.fingerprint,
            "solver_snapshot_timestamp": self.snapshot_timestamp,
        }


def target_graph_fingerprint(graph: nx.Graph) -> str:
    """Target graph 의 결정적 지문.

    노드 집합과 엣지 집합 전체를 해싱한다. 같은 solver_name 이라도
    결함이 달라지면 지문이 달라진다.
    """
    nodes = sorted(graph.nodes())
    edges = sorted(tuple(sorted(e)) for e in graph.edges())
    blob = repr((nodes, edges)).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return f"n{len(nodes)}e{len(edges)}_{digest}"


def build_ideal_target(topology: str, size: int, label: str) -> TargetGraph:
    """결함 없는 이상적 target graph 를 만든다.

    Args:
        topology: ``"pegasus"`` 또는 ``"zephyr"``.
        size: topology shape (P16 이면 16, Z12 이면 12).
        label: 결과에 기록할 label.

    Returns:
        :class:`TargetGraph`.

    Raises:
        ValueError: 지원하지 않는 topology 인 경우.
    """
    import dwave_networkx as dnx

    if topology == "pegasus":
        graph = dnx.pegasus_graph(size)
    elif topology == "zephyr":
        graph = dnx.zephyr_graph(size)
    else:
        raise ValueError(f"지원하지 않는 topology: {topology!r}")

    target = TargetGraph(
        label=label,
        topology=topology,
        topology_shape=size,
        mode=TargetMode.IDEAL,
        graph=graph,
        solver_id=None,
        graph_id=f"ideal_{topology}_{size}",
        metadata={"ideal": True},
    )
    return target


def build_ideal_targets(cfg: dict) -> list[TargetGraph]:
    """설정의 모든 이상적 target 을 만든다."""
    return [
        build_ideal_target(spec["topology"], int(spec["size"]), spec["label"])
        for spec in cfg["embedding"]["ideal_targets"]
    ]


@dataclass
class QPUSelection:
    """실제 QPU 선택 결과."""

    status: Status
    target: TargetGraph | None = None
    message: str = ""
    available_solvers: list[str] = field(default_factory=list)


def select_actual_qpu(topology_key: str, cfg: dict) -> QPUSelection:
    """실제 QPU 를 선택하고 topology/shape 를 검증한다.

    선택 규칙(명세 14절):

    1. ``solver_name`` 이 있으면 해당 solver 를 사용한다.
    2. 실제 topology 와 shape 가 요구사항과 다르면 ``TOPOLOGY_MISMATCH``.
    3. ``solver_name`` 이 없으면 접근 가능한 QPU 목록을 조회한다.
    4. 정확히 하나이면 자동 선택한다.
    5. 0개이면 ``NO_QPU_ACCESS``.
    6. 둘 이상이면 임의 선택하지 않는다.
    7. solver 목록을 남기고 ``AMBIGUOUS_QPU_SELECTION``.

    Args:
        topology_key: ``qa.solvers`` 아래의 key (``"pegasus"`` 또는 ``"zephyr"``).
        cfg: 전체 설정 dict.

    Returns:
        :class:`QPUSelection`. QPU 접근이 불가능해도 예외를 던지지 않는다.
    """
    import datetime as _dt

    spec = cfg["qa"]["solvers"][topology_key]
    required_topology = spec["required_topology"]
    required_shape = int(spec["required_shape"])

    try:
        from dwave.cloud import Client
    except Exception as exc:  # pragma: no cover - 환경 의존
        return QPUSelection(
            status=Status.NO_QPU_ACCESS, message=f"dwave-cloud-client import 실패: {exc}"
        )

    try:
        client = Client.from_config(profile=cfg["qa"].get("profile"))
    except Exception as exc:
        return QPUSelection(
            status=Status.NO_QPU_ACCESS,
            message=f"D-Wave 설정을 로드할 수 없다: {exc}",
        )

    try:
        with client:
            candidates = client.get_solvers(qpu=True, topology__type=required_topology)
            names = [s.id for s in candidates]

            if spec.get("solver_name"):
                chosen = next((s for s in candidates if s.id == spec["solver_name"]), None)
                if chosen is None:
                    try:
                        chosen = client.get_solver(name=spec["solver_name"])
                    except Exception as exc:
                        return QPUSelection(
                            status=Status.NO_QPU_ACCESS,
                            message=f"지정한 solver {spec['solver_name']!r} 에 접근할 수 없다: {exc}",
                            available_solvers=names,
                        )
            elif len(candidates) == 0:
                return QPUSelection(
                    status=Status.NO_QPU_ACCESS,
                    message=f"{required_topology} topology QPU 에 접근할 수 없다",
                    available_solvers=names,
                )
            elif len(candidates) == 1:
                chosen = candidates[0]
            else:
                return QPUSelection(
                    status=Status.AMBIGUOUS_QPU_SELECTION,
                    message=(
                        f"{required_topology} QPU 가 {len(candidates)}개 접근 가능하다. "
                        "임의로 선택하지 않는다. solver_name 을 설정하라"
                    ),
                    available_solvers=names,
                )

            properties = chosen.properties
            actual_topology = properties["topology"]["type"]
            actual_shape = properties["topology"]["shape"]

            if actual_topology != required_topology or int(actual_shape[0]) != required_shape:
                return QPUSelection(
                    status=Status.TOPOLOGY_MISMATCH,
                    message=(
                        f"solver {chosen.id}: topology {actual_topology}{actual_shape} 가 "
                        f"요구사항 {required_topology}[{required_shape}] 와 다르다"
                    ),
                    available_solvers=names,
                )

            graph = nx.Graph()
            graph.add_nodes_from(properties["qubits"])
            graph.add_edges_from(properties["couplers"])

            target = TargetGraph(
                label=f"actual_{required_topology}_{chosen.id}",
                topology=actual_topology,
                topology_shape=int(actual_shape[0]),
                mode=TargetMode.ACTUAL,
                graph=graph,
                solver_id=chosen.id,
                chip_id=properties.get("chip_id"),
                snapshot_timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                metadata={"topology_shape_full": list(actual_shape)},
            )
            target.graph_id = target.fingerprint
            return QPUSelection(
                status=Status.SUCCESS, target=target, available_solvers=names
            )
    except Exception as exc:  # pragma: no cover - 환경 의존
        return QPUSelection(status=Status.NO_QPU_ACCESS, message=f"QPU 조회 실패: {exc}")


def list_accessible_qpus(
    profile: str | None = None,
) -> tuple[Status, list[dict[str, Any]], str]:
    """접근 가능한 QPU 목록을 조회한다 (notebook 05 preflight 용).

    Args:
        profile: ``dwave.conf`` 의 프로필 이름. ``None`` 이면 기본 프로필을 쓴다.
            환경변수 ``DWAVE_API_TOKEN`` 이 설정되어 있으면 그쪽이 우선한다.

    Returns:
        ``(status, solver_info_list, message)``. QPU 접근이 불가능해도
        예외를 던지지 않는다.
    """
    try:
        from dwave.cloud import Client
    except Exception as exc:
        return Status.NO_QPU_ACCESS, [], f"dwave-cloud-client import 실패: {exc}"
    try:
        with Client.from_config(profile=profile) as client:
            info = []
            for solver in client.get_solvers(qpu=True):
                topology = solver.properties.get("topology", {})
                info.append(
                    {
                        "solver_id": solver.id,
                        "chip_id": solver.properties.get("chip_id"),
                        "topology": topology.get("type"),
                        "topology_shape": topology.get("shape"),
                        "num_qubits": len(solver.properties.get("qubits", [])),
                        "num_couplers": len(solver.properties.get("couplers", [])),
                    }
                )
            if not info:
                return Status.NO_QPU_ACCESS, [], "접근 가능한 QPU 가 없다"
            return Status.SUCCESS, info, f"{len(info)}개 QPU 접근 가능"
    except Exception as exc:
        return Status.NO_QPU_ACCESS, [], f"QPU 목록 조회 실패: {exc}"
