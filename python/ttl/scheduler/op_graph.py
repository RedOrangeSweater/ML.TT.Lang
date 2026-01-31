# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation graph: internal representation for scheduler (Phase 3).

Nodes are ops (load, store, elementwise, ...); edges are data dependencies.
Optional resource annotation (color) per node: NOC, SFPU, FPU, etc.
"""

from __future__ import annotations


from pydantic import BaseModel, Field


# Thread type to op_type/resource mapping (avoids long if-chains in build_op_graph_from_threads)
_THREAD_TYPE_TO_OP: dict[str, tuple[str, str]] = {
    "compute": ("elementwise", "SFPU"),
    "datamovement": ("dm", "NOC"),
    "noc": ("dm", "NOC"),
}


def _op_type_for_dm_index(index: int, total_dm: int) -> str:
    """Map DM index to load/dm/store."""
    if total_dm <= 0:
        return "dm"
    if index == 0:
        return "load"
    if index == total_dm - 1:
        return "store"
    return "dm"


class OpNode(BaseModel):
    """Single node in the op graph."""

    id: str
    op_type: str
    resource: str | None = None
    predecessors: list[str] = Field(default_factory=list)
    successors: list[str] = Field(default_factory=list)

    def __repr__(self) -> str:
        return f"OpNode(id={self.id!r}, op_type={self.op_type!r})"


class OpGraph(BaseModel):
    """DAG of operations with data dependencies."""

    nodes: dict[str, OpNode] = Field(default_factory=dict)

    def add_node(self, node: OpNode) -> OpGraph:
        """Return new graph with node added (immutable)."""
        return self.model_copy(update={"nodes": {**self.nodes, node.id: node}})

    def add_edge(self, from_id: str, to_id: str) -> OpGraph:
        """Return new graph with edge added if both nodes exist."""
        if from_id not in self.nodes or to_id not in self.nodes:
            return self
        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]
        new_from = from_node.model_copy(
            update={"successors": [*from_node.successors, to_id]}
        )
        new_to = to_node.model_copy(
            update={"predecessors": [*to_node.predecessors, from_id]}
        )
        return self.model_copy(
            update={"nodes": {**self.nodes, from_id: new_from, to_id: new_to}}
        )

    def node_ids_in_order(self) -> list[str]:
        """Topological order (simplified: same as insertion)."""
        return list(self.nodes.keys())

    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON (list of nodes)."""
        return {"nodes": [n.model_dump() for n in self.nodes.values()]}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> OpGraph:
        """Build from JSON dict (nodes as list)."""
        nodes = {n["id"]: OpNode(**n) for n in data.get("nodes", [])}
        return cls(nodes=nodes)


class OpGraphSchema(BaseModel):
    """JSON shape for op_graph (nodes as list). Used by SchedulerInput."""

    nodes: list[OpNode] = Field(default_factory=list)


def build_op_graph_from_threads(
    thread_infos: list[tuple[str, str]],
) -> OpGraph:
    """
    Build op graph from list of (thread_name, thread_type).

    thread_type is "compute" or "datamovement"/"noc".
    RCW: first DM = load, last DM = store, compute = elementwise.
    Edges: load -> elementwise -> store.
    """
    compute_ids: list[str] = []
    dm_ids: list[str] = []
    for name, kt in thread_infos:
        if kt == "compute":
            compute_ids.append(name)
        else:
            dm_ids.append(name)

    graph = OpGraph()
    for i, nid in enumerate(dm_ids):
        op_type = _op_type_for_dm_index(i, len(dm_ids))
        graph = graph.add_node(OpNode(id=nid, op_type=op_type, resource="NOC"))
    for nid in compute_ids:
        graph = graph.add_node(OpNode(id=nid, op_type="elementwise", resource="SFPU"))

    if dm_ids and compute_ids:
        graph = graph.add_edge(dm_ids[0], compute_ids[0])
    if compute_ids and len(dm_ids) >= 2:
        graph = graph.add_edge(compute_ids[0], dm_ids[-1])
    return graph
