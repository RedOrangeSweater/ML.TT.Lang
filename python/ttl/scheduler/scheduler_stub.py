# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler stub: produces plan that reproduces current RCW behavior (Phase 4).

Input: op graph + topology + optional program_config.
Output: plan (placement of each op on core); stub always assigns all to (0,0) for grid=(1,1).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .op_graph import OpGraph
from .topology import Topology


class SchedulePlan(BaseModel):
    """Placement of each op (node id) on core (col, row). JSON uses list for coords."""

    placement: dict[str, list[int]] = Field(default_factory=dict)
    grid_cols: int = Field(..., ge=1)
    grid_rows: int = Field(..., ge=1)

    def core_for(self, node_id: str) -> tuple[int, int]:
        """Return (col, row) for node; (0, 0) if not placed."""
        coords = self.placement.get(node_id, [0, 0])
        return (coords[0], coords[1]) if len(coords) >= 2 else (0, 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON."""
        return self.model_dump()


def schedule_stub(
    graph: OpGraph,
    topology: Topology,
    program_config: dict | None = None,
) -> SchedulePlan:
    """
    Stub scheduler: reproduces current RCW behavior.

    All ops placed on core (0, 0). Does not use program_config (Phase 2).
    """
    _ = program_config
    placement = {nid: [0, 0] for nid in graph.node_ids_in_order()}
    return SchedulePlan(
        placement=placement,
        grid_cols=topology.grid_cols,
        grid_rows=topology.grid_rows,
    )
