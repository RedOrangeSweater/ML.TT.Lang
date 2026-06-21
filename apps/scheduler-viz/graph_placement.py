# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Graph-based placement search for scheduler-viz (Toy domain, small instances).

Brute-force optimal placement for small op graphs on grid topology.
Used by dijkstra/a_star policies when the remaining search space is tiny.
"""

from __future__ import annotations

from itertools import product
from typing import Any


def _edge_list(graph: dict[str, Any]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for node in graph["nodes"]:
        for succ in node.get("successors", []):
            edges.append((node["id"], succ))
    return edges


def _core_coords(grid_cols: int, grid_rows: int) -> list[tuple[int, int]]:
    return [(c, r) for r in range(grid_rows) for c in range(grid_cols)]


def _manhattan(c1: int, r1: int, c2: int, r2: int) -> int:
    return abs(c1 - c2) + abs(r1 - r2)


def placement_edge_cost(
    placement: dict[str, tuple[int, int]],
    graph: dict[str, Any],
) -> float:
    """Sum of Manhattan distances over dataflow edges (lower is better)."""
    total = 0.0
    for u, v in _edge_list(graph):
        if u not in placement or v not in placement:
            continue
        c1, r1 = placement[u]
        c2, r2 = placement[v]
        total += float(_manhattan(c1, r1, c2, r2))
    return total


def optimal_placement_bruteforce(
    graph: dict[str, Any],
    topology: dict[str, Any],
    node_ids: list[str],
) -> tuple[dict[str, tuple[int, int]], float]:
    """
    Exhaustive search over core assignments (Toy-sized graphs only).

    Returns (best_placement, edge_cost).
    """
    coords = _core_coords(topology["grid_cols"], topology["grid_rows"])
    if not node_ids or not coords:
        return {}, 0.0

    best_placement: dict[str, tuple[int, int]] = {}
    best_cost = float("inf")

    for assignment in product(range(len(coords)), repeat=len(node_ids)):
        placement = {node_ids[i]: coords[assignment[i]] for i in range(len(node_ids))}
        cost = placement_edge_cost(placement, graph)
        if cost < best_cost:
            best_cost = cost
            best_placement = placement

    return best_placement, best_cost


def _complete_optimal_with_prefix(
    prefix: dict[str, tuple[int, int]],
    remaining_ids: list[str],
    graph: dict[str, Any],
    topology: dict[str, Any],
) -> float:
    """Min edge cost completing placement for remaining_ids given fixed prefix."""
    if not remaining_ids:
        return placement_edge_cost(prefix, graph)

    coords = _core_coords(topology["grid_cols"], topology["grid_rows"])
    best_cost = float("inf")
    for assignment in product(range(len(coords)), repeat=len(remaining_ids)):
        placement = dict(prefix)
        for i, nid in enumerate(remaining_ids):
            placement[nid] = coords[assignment[i]]
        cost = placement_edge_cost(placement, graph)
        if cost < best_cost:
            best_cost = cost
    return best_cost


def optimal_core_for_next_node(
    partial: dict[str, tuple[int, int]],
    node_ids: list[str],
    next_idx: int,
    graph: dict[str, Any],
    topology: dict[str, Any],
) -> int:
    """
    Pick core index for node_ids[next_idx] minimizing final edge cost.

    Completes remaining nodes via brute force (Toy instances only).
    """
    coords = _core_coords(topology["grid_cols"], topology["grid_rows"])
    current_id = node_ids[next_idx]
    tail_ids = node_ids[next_idx + 1 :]

    best_core = 0
    best_cost = float("inf")

    for core_idx, coord in enumerate(coords):
        prefix = dict(partial)
        prefix[current_id] = coord
        cost = _complete_optimal_with_prefix(prefix, tail_ids, graph, topology)
        if cost < best_cost:
            best_cost = cost
            best_core = core_idx

    return best_core


def a_star_core_for_next_node(
    partial: dict[str, tuple[int, int]],
    node_ids: list[str],
    next_idx: int,
    graph: dict[str, Any],
    topology: dict[str, Any],
) -> int:
    """
    A* placement policy for Toy graphs.

    Uses optimal completion as heuristic; equivalent to dijkstra policy here
    because the cost model is static edge Manhattan sum on small instances.
    """
    return optimal_core_for_next_node(partial, node_ids, next_idx, graph, topology)
