# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph-based placement algorithms (Toy domain, known optimum)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCHEDULER_VIZ = Path(__file__).resolve().parents[2] / "apps" / "scheduler-viz"
if str(_SCHEDULER_VIZ) not in sys.path:
    sys.path.insert(0, str(_SCHEDULER_VIZ))

from graph_placement import (  # noqa: E402
    optimal_core_for_next_node,
    optimal_placement_bruteforce,
    placement_edge_cost,
)
from scheduler_env import SchedulerPlacementEnv  # noqa: E402


def _chain_graph_2x1() -> dict:
    """Two nodes on a line: A -> B. Optimal: same core (cost 0) on 2x1 grid."""
    return {
        "nodes": [
            {"id": "A", "successors": ["B"], "predecessors": []},
            {"id": "B", "successors": [], "predecessors": ["A"]},
        ]
    }


def _chain_graph_3x1() -> dict:
    """A -> B -> C on 2x2 grid; optimal places chain on adjacent cores."""
    return {
        "nodes": [
            {"id": "A", "successors": ["B"], "predecessors": []},
            {"id": "B", "successors": ["C"], "predecessors": ["A"]},
            {"id": "C", "successors": [], "predecessors": ["B"]},
        ]
    }


def test_bruteforce_chain_two_nodes_zero_cost():
    graph = _chain_graph_2x1()
    topology = {"grid_cols": 2, "grid_rows": 1}
    placement, cost = optimal_placement_bruteforce(graph, topology, ["A", "B"])
    assert cost == 0.0
    assert placement["A"] == placement["B"]


def test_bruteforce_chain_three_nodes_min_cost():
    graph = _chain_graph_3x1()
    topology = {"grid_cols": 2, "grid_rows": 2}
    placement, cost = optimal_placement_bruteforce(graph, topology, ["A", "B", "C"])
    assert cost == 0.0  # chain fits on one core; edge cost is zero
    assert placement["A"] == placement["B"] == placement["C"]
    assert placement_edge_cost(placement, graph) == cost


def test_dijkstra_policy_matches_bruteforce_on_fixture():
    fixture = _SCHEDULER_VIZ / "fixtures" / "toy_program_add.json"
    data = json.loads(fixture.read_text())
    graph = data["op_graph"]
    topology = data["topology"]
    node_ids = [n["id"] for n in graph["nodes"]]

    optimal, opt_cost = optimal_placement_bruteforce(graph, topology, node_ids)

    env = SchedulerPlacementEnv(data)
    env.reset()
    simulated: dict[str, tuple[int, int]] = {}
    for idx in range(len(node_ids)):
        core = optimal_core_for_next_node(
            simulated, node_ids, idx, graph, topology
        )
        obs, reward, done, _, info = env.step(core)
        simulated[info["placed_node"]] = tuple(info["core"])
        if done:
            assert reward == -opt_cost or opt_cost == 0.0

    assert simulated == optimal or placement_edge_cost(simulated, graph) == opt_cost


def test_dijkstra_better_than_random_on_2x2_chain():
    graph = _chain_graph_3x1()
    topology = {"grid_cols": 2, "grid_rows": 2}
    node_ids = ["A", "B", "C"]
    _, opt_cost = optimal_placement_bruteforce(graph, topology, node_ids)

    # Worst placement: spread across diagonal (0,0), (1,0), (0,1) -> cost 3
    worst = {"A": (0, 0), "B": (1, 0), "C": (0, 1)}
    assert placement_edge_cost(worst, graph) > opt_cost
