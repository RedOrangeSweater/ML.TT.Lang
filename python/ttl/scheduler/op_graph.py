# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation graph: internal representation for scheduler (Phase 3).

Nodes are ops (load, store, elementwise, ...); edges are data dependencies.
Optional resource annotation (color) per node: NOC, SFPU, FPU, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpNode:
    """Single node in the op graph."""

    id: str  # thread/kernel name
    op_type: str  # "load" | "store" | "elementwise" | "dm" (generic)
    resource: str | None = None  # optional "NOC" | "SFPU" | "FPU" | ...
    predecessors: list[str] = field(default_factory=list)  # node ids
    successors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"OpNode(id={self.id!r}, op_type={self.op_type!r})"


@dataclass
class OpGraph:
    """DAG of operations with data dependencies."""

    nodes: dict[str, OpNode] = field(default_factory=dict)

    def add_node(self, node: OpNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].successors.append(to_id)
            self.nodes[to_id].predecessors.append(from_id)

    def node_ids_in_order(self) -> list[str]:
        """Return node ids in topological order (simplified: same as insertion)."""
        return list(self.nodes.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tests/logging."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "op_type": n.op_type,
                    "resource": n.resource,
                    "predecessors": n.predecessors,
                    "successors": n.successors,
                }
                for n in self.nodes.values()
            ]
        }


def build_op_graph_from_threads(
    thread_infos: list[tuple[str, str]],
) -> OpGraph:
    """
    Build op graph from list of (thread_name, thread_type).

    thread_type is "compute" or "datamovement".
    For RCW (1 compute + 2 DM): first DM = load, second DM = store, compute = elementwise.
    Edges: load -> elementwise -> store.
    """
    graph = OpGraph()
    compute_ids: list[str] = []
    dm_ids: list[str] = []

    for name, kt in thread_infos:
        if kt == "compute":
            compute_ids.append(name)
        elif kt == "datamovement" or kt == "noc":
            dm_ids.append(name)
        else:
            dm_ids.append(name)

    # Assign op types: for RCW, first dm = load, last dm = store, compute = elementwise
    for i, nid in enumerate(dm_ids):
        op_type = "load" if i == 0 else "store" if i == len(dm_ids) - 1 else "dm"
        resource = "NOC"
        graph.add_node(OpNode(id=nid, op_type=op_type, resource=resource))

    for nid in compute_ids:
        graph.add_node(OpNode(id=nid, op_type="elementwise", resource="SFPU"))

    # Edges: load -> elementwise -> store (for 1 compute, 2 dm)
    if len(dm_ids) >= 1 and len(compute_ids) >= 1:
        graph.add_edge(dm_ids[0], compute_ids[0])
    if len(compute_ids) >= 1 and len(dm_ids) >= 2:
        graph.add_edge(compute_ids[0], dm_ids[-1])

    return graph
