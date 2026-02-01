# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Export scheduler input (op graph, topology, plan) to JSON-serializable dict.

For use by scheduler viz UI and Gymnasium environment (doc 16).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .op_graph import OpGraphSchema, build_op_graph_from_threads
from .scheduler_stub import SchedulePlan, schedule_stub
from .topology import Topology, build_topology_from_grid


class SchedulerInput(BaseModel):
    """Unified scheduler input: op_graph, topology, plan. Validates and serializes JSON."""

    op_graph: OpGraphSchema
    topology: Topology
    plan: SchedulePlan

    def model_dump(self, **kwargs: object) -> dict[str, object]:
        """Serialize for JSON (same shape as existing fixtures)."""
        return super().model_dump(**kwargs)


def export_scheduler_input(
    thread_infos: list[tuple[str, str]],
    grid: tuple[int, ...] | list[int],
    program_config: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build op graph, topology, and stub plan; return JSON-serializable dict.

    Keys: "op_graph", "topology", "plan". Used by scheduler viz UI and Gym env.
    """
    graph = build_op_graph_from_threads(thread_infos)
    topology = build_topology_from_grid(grid)
    plan = schedule_stub(graph, topology, program_config)
    schema = OpGraphSchema(nodes=list(graph.nodes.values()))
    inp = SchedulerInput(op_graph=schema, topology=topology, plan=plan)
    return inp.model_dump()


def export_scheduler_input_to_json(
    thread_infos: list[tuple[str, str]],
    grid: tuple[int, ...] | list[int],
    path: str | Path,
    program_config: dict[str, object] | None = None,
    indent: int = 2,
) -> None:
    """Export scheduler input to a JSON file."""
    data = export_scheduler_input(thread_infos, grid, program_config)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
