# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Toy backend scheduler stub: builds minimal plan from AbstractEngineConfig (doc 19/20).

Uses graph_toy_generator and/or topology_graph to produce node list, maps nodes to 1xN grid,
returns SchedulePlan with stub placement for scheduler-viz and Gym (doc 16).
"""

from __future__ import annotations

from .config import AbstractEngineConfig
from .scheduler_stub import SchedulePlan


def schedule_toy_stub(config: AbstractEngineConfig) -> SchedulePlan:
    """
    Stub scheduler for Toy backend: node list from config, placement on 1xN grid.

    Uses topology_graph.nodes if present, else synthetic node ids from graph_toy_generator.
    Returns SchedulePlan compatible with scheduler export (grid_cols=1, grid_rows=len(nodes)).
    """
    if config.topology_graph is not None and config.topology_graph.nodes:
        node_ids = list(config.topology_graph.nodes)
    elif config.graph_toy_generator is not None:
        n = config.graph_toy_generator.num_nodes
        node_ids = [f"node_{i}" for i in range(n)]
    else:
        node_ids = ["node_0"]
    n_nodes = len(node_ids)
    placement = {nid: [0, i] for i, nid in enumerate(node_ids)}
    return SchedulePlan(
        placement=placement,
        grid_cols=1,
        grid_rows=n_nodes,
    )
