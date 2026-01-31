# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler stub: produces plan that reproduces current RCW behavior (Phase 4).

Input: op graph + topology + optional program_config.
Output: plan (placement of each op on core); stub always assigns all to (0,0) for grid=(1,1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .op_graph import OpGraph
from .topology import Topology


@dataclass
class SchedulePlan:
    """Placement of each op (node id) on core (col, row)."""

    placement: dict[str, tuple[int, int]]  # node_id -> (col, row)
    grid_cols: int
    grid_rows: int

    def core_for(self, node_id: str) -> tuple[int, int]:
        return self.placement.get(node_id, (0, 0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for export (e.g. JSON for scheduler viz). Placement as node_id -> [col, row]."""
        return {
            "placement": {nid: [c, r] for nid, (c, r) in self.placement.items()},
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
        }


def schedule_stub(
    graph: OpGraph,
    topology: Topology,
    program_config: dict | None = None,
) -> SchedulePlan:
    """
    Stub scheduler: reproduces current RCW behavior.

    All ops placed on core (0, 0) for single-core grid; grid from topology.
    Does not use program_config for decisions (Phase 2).
    """
    _ = program_config
    placement: dict[str, tuple[int, int]] = {}
    for nid in graph.node_ids_in_order():
        placement[nid] = (0, 0)
    return SchedulePlan(
        placement=placement,
        grid_cols=topology.grid_cols,
        grid_rows=topology.grid_rows,
    )
