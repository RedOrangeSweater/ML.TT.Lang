# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gymnasium environment for scheduler placement (doc 16).

Phase 1 (placement, current): state = op graph + topology + placement; action = core index;
reward = stub (minus sum of edge distances). Phase 2 (planned): after placement, run dynamics
simulator (doc 15 or simplified model); reward and comparison from trajectory metrics
(utilization, latency, bottlenecks), not static placement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

try:
    from models import SchedulerInputPayload
except ImportError:
    SchedulerInputPayload = None  # type: ignore[misc, assignment]


def _scheduler_input_to_dict(
    scheduler_input: dict[str, Any] | str | Path | Any,
) -> dict[str, Any]:
    """Normalize scheduler input to dict (op_graph, topology, plan). Accepts model, dict, or path."""
    if SchedulerInputPayload is not None and isinstance(
        scheduler_input, SchedulerInputPayload
    ):
        return scheduler_input.to_env_dict()
    if isinstance(scheduler_input, dict):
        return scheduler_input
    path = Path(scheduler_input)
    with path.open() as f:
        return json.load(f)


def _node_order(graph: dict[str, Any]) -> list[str]:
    """Topological order: nodes by dependencies (simplified: use list order)."""
    nodes = [n["id"] for n in graph["nodes"]]
    return nodes


def _core_coords(grid_cols: int, grid_rows: int) -> list[tuple[int, int]]:
    """Row-major list of (col, row)."""
    return [(c, r) for r in range(grid_rows) for c in range(grid_cols)]


def _edge_list(graph: dict[str, Any]) -> list[tuple[str, str]]:
    """List of (from_id, to_id) edges."""
    edges: list[tuple[str, str]] = []
    for n in graph["nodes"]:
        for succ in n.get("successors", []):
            edges.append((n["id"], succ))
    return edges


def _manhattan(c1: int, r1: int, c2: int, r2: int) -> int:
    """Manhattan distance between (c1, r1) and (c2, r2)."""
    return abs(c1 - c2) + abs(r1 - r2)


def _sum_edge_manhattan(
    edges: list[tuple[str, str]],
    placement: dict[str, tuple[int, int]],
) -> float:
    """Sum of Manhattan distances over edges; 0 if any endpoint unplaced."""
    total = 0.0
    for u, v in edges:
        if u not in placement or v not in placement:
            return 0.0
        c1, r1 = placement[u]
        c2, r2 = placement[v]
        total += _manhattan(c1, r1, c2, r2)
    return total


def _reward_stub(
    graph: dict[str, Any],
    topology: dict[str, Any],
    placement: dict[str, tuple[int, int]],
) -> float:
    """Stub reward: minus sum of Manhattan distances over edges (lower comm cost = higher reward)."""
    edges = _edge_list(graph)
    if not edges:
        return 0.0
    return -float(_sum_edge_manhattan(edges, placement))


def reward_from_simulator_stub(
    graph: dict[str, Any],
    topology: dict[str, Any],
    placement: dict[str, tuple[int, int]],
) -> float:
    """
    Placeholder for tt-sim / emulator reward (doc 15, 16).

    Replace with a call to tt-sim or salabim emulator to get latency/throughput
    and return a scalar (e.g. minus latency or plus utilization).
    """
    return _reward_stub(graph, topology, placement)


class SchedulerPlacementEnv(gym.Env[dict[str, Any], int]):
    """
    Place op graph nodes onto topology cores sequentially.

    Observation: dict with "placement" (n_nodes x 2, -1 = unplaced), "next_node" (index).
    Action: core index (Discrete(n_cores)).
    Reward: stub = -sum of edge distances when done.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scheduler_input: dict[str, Any] | str | Path | Any,
        render_mode: str | None = None,
        reward_fn: (
            Callable[
                [dict[str, Any], dict[str, Any], dict[str, tuple[int, int]]],
                float,
            ]
            | None
        ) = None,
    ):
        super().__init__()
        self._reward_fn = reward_fn or _reward_stub
        data = _scheduler_input_to_dict(scheduler_input)
        self._graph = data["op_graph"]
        self._topology = data["topology"]
        self._node_ids = _node_order(self._graph)
        self._n_nodes = len(self._node_ids)
        self._coords = _core_coords(
            self._topology["grid_cols"],
            self._topology["grid_rows"],
        )
        self._n_cores = len(self._coords)
        self._placement: dict[str, tuple[int, int]] = {}
        self._next_idx = 0

        # Observation: placement matrix (n_nodes, 2) with -1 for unplaced; next_node index
        max_coord = max(
            self._topology["grid_cols"] - 1,
            self._topology["grid_rows"] - 1,
            0,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "placement": gym.spaces.Box(
                    low=-1,
                    high=max_coord,
                    shape=(self._n_nodes, 2),
                    dtype=np.int32,
                ),
                "next_node": gym.spaces.Discrete(self._n_nodes + 1),
            }
        )
        self.action_space = gym.spaces.Discrete(self._n_cores)
        self.render_mode = render_mode

    def _obs(self) -> dict[str, Any]:
        pl = np.full((self._n_nodes, 2), -1, dtype=np.int32)
        for i, nid in enumerate(self._node_ids):
            if nid in self._placement:
                c, r = self._placement[nid]
                pl[i, 0], pl[i, 1] = c, r
        return {
            "placement": pl,
            "next_node": self._next_idx,
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._placement = {}
        self._next_idx = 0
        return self._obs(), {
            "placement": dict(self._placement),
            "graph": self._graph,
            "topology": self._topology,
        }

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._next_idx >= self._n_nodes:
            return self._obs(), 0.0, True, False, {}
        if not 0 <= action < self._n_cores:
            return self._obs(), -10.0, False, False, {"error": "invalid core"}
        nid = self._node_ids[self._next_idx]
        self._placement[nid] = self._coords[action]
        self._next_idx += 1
        terminated = self._next_idx >= self._n_nodes
        reward = 0.0
        if terminated:
            reward = self._reward_fn(self._graph, self._topology, self._placement)
        info = {
            "placement": dict(self._placement),
            "placed_node": nid,
            "core": self._coords[action],
        }
        return self._obs(), reward, terminated, False, info

    def get_full_state(self) -> dict[str, Any]:
        """Return full state for UI: graph, topology, placement, plan dict."""
        placement_list = {nid: [c, r] for nid, (c, r) in self._placement.items()}
        return {
            "op_graph": self._graph,
            "topology": self._topology,
            "plan": {
                "placement": placement_list,
                "grid_cols": self._topology["grid_cols"],
                "grid_rows": self._topology["grid_rows"],
            },
        }


def get_action_rcw(env: SchedulerPlacementEnv) -> int:
    """RCW preset: always place on core 0 (first core)."""
    return 0


def get_action_random(env: SchedulerPlacementEnv) -> int:
    """Random policy: sample from action space."""
    return int(env.action_space.sample())
