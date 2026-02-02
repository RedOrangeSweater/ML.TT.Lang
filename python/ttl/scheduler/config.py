# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Abstract engine config: Pydantic models and loader for scheduler/Toy backend config.

Variant A (doc 20): load from JSON (or YAML if available); validate via Pydantic.
Backend, graph spec, topology spec, objective. See docs/sdlc/00_Main/00_Ideas/20_nickel_mlir_config_abstract_engine.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..constants import Objective

BackendType = Literal["tenstorrent", "toy_shops", "toy_bakeries"]


class GraphSourceSpec(BaseModel):
    """Graph from external source (Program/IR or generator path)."""

    source: Literal["program", "generator"] = "program"
    path: str | None = Field(default=None, description="Path to IR or generator script")
    program_hash: int | None = Field(
        default=None, description="Program hash when source=program"
    )


class ToyGraphGeneratorSpec(BaseModel):
    """Parameters for Toy domain graph generator (shops/bakeries)."""

    num_nodes: int = Field(
        ..., ge=1, description="Number of nodes (factories/stores/bakeries)"
    )
    product_types: list[str] = Field(
        default_factory=list, description="Product/operation types"
    )
    bom: dict[str, list[str]] = Field(
        default_factory=dict,
        description="BOM: product -> list of input product types",
    )


class TopologyGridSpec(BaseModel):
    """Topology as rectangular grid (Tenstorrent-style)."""

    grid_cols: int = Field(..., ge=1)
    grid_rows: int = Field(..., ge=1)


class TopologyEdgeSpec(BaseModel):
    """Single edge in topology graph (Toy: road between nodes)."""

    from_id: str
    to_id: str
    latency: float | None = Field(default=None)
    capacity: float | None = Field(default=None)


class TopologyGraphSpec(BaseModel):
    """Topology as explicit graph (nodes + edges)."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[TopologyEdgeSpec] = Field(default_factory=list)


class SimulatorOptions(BaseModel):
    """Optional simulator backend and parameters."""

    backend: Literal["salabim", "stub"] | None = Field(default=None)
    params: dict[str, object] = Field(default_factory=dict)


class PlannerOptions(BaseModel):
    """Optional planner algorithm and limits."""

    algorithm: str | None = Field(default=None, description="e.g. rcw, dijkstra, mcts")
    max_iterations: int | None = Field(default=None, ge=1)


class AbstractEngineConfig(BaseModel):
    """
    Configuration for abstract engine (scheduler/Toy backend).

    Load from JSON (Nickel export target). Validated via Pydantic.
    See docs/sdlc/00_Main/00_Ideas/20_nickel_mlir_config_abstract_engine.md.
    """

    backend: BackendType = Field(
        ..., description="tenstorrent | toy_shops | toy_bakeries"
    )
    objective: Objective = Field(default=Objective.LATENCY)

    # Graph: either source ref or Toy generator params
    graph_source: GraphSourceSpec | None = Field(default=None)
    graph_toy_generator: ToyGraphGeneratorSpec | None = Field(default=None)

    # Topology: either grid (Tenstorrent) or explicit graph (Toy)
    topology_grid: TopologyGridSpec | None = Field(default=None)
    topology_graph: TopologyGraphSpec | None = Field(default=None)

    simulator_options: SimulatorOptions | None = Field(default=None)
    planner_options: PlannerOptions | None = Field(default=None)

    def get_topology_grid(self) -> tuple[int, int] | None:
        """Return (grid_cols, grid_rows) if topology is grid."""
        if self.topology_grid is None:
            return None
        return (self.topology_grid.grid_cols, self.topology_grid.grid_rows)


def load_abstract_engine_config(path: str | Path) -> AbstractEngineConfig:
    """
    Load abstract engine config from JSON file.

    Path can be .json or .yaml; YAML requires PyYAML. Validates via Pydantic.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(raw)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .json or .yaml")

    return AbstractEngineConfig.model_validate(data)


def validate_topology_connectivity(config: AbstractEngineConfig) -> bool:
    """
    Check that topology graph is connected (if topology_graph is used).

    Returns True if grid or if graph has at most one node or all nodes
    are reachable from the first. Raises ValueError on invalid config.
    """
    if config.topology_graph is None:
        return True
    nodes = set(config.topology_graph.nodes)
    if len(nodes) <= 1:
        return True
    edges = config.topology_graph.edges
    # Build adjacency (undirected for connectivity)
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        if e.from_id in adj:
            adj[e.from_id].add(e.to_id)
        if e.to_id in adj:
            adj[e.to_id].add(e.from_id)
    # BFS from first node
    start = config.topology_graph.nodes[0]
    seen = {start}
    stack = [start]
    while stack:
        n = stack.pop()
        for neighbor in adj.get(n, []):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if seen != nodes:
        raise ValueError(
            f"Topology graph is not connected: {len(seen)}/{len(nodes)} nodes reachable"
        )
    return True
