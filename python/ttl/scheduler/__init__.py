# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler package (Graph layer): op graph, topology, scheduler stub, abstract engine config.

Graph layer: OpGraph, SchedulePlan, Topology. One focus: operations and
placement plan; R/C/W is an engine placement pattern, not the API axis.
Phase 3-4: build op graph from program threads, topology from grid,
scheduler stub produces plan that reproduces current RCW behavior.
Abstract engine config (doc 20): load from JSON/YAML for backend/graph/topology/objective.
"""

from .config import (
    AbstractEngineConfig,
    GraphSourceSpec,
    PlannerOptions,
    SimulatorOptions,
    TopologyEdgeSpec,
    TopologyGraphSpec,
    TopologyGridSpec,
    ToyGraphGeneratorSpec,
    load_abstract_engine_config,
    validate_topology_connectivity,
)
from .export import (
    SchedulerInput,
    export_scheduler_input,
    export_scheduler_input_to_json,
)
from .op_graph import OpGraph, OpGraphSchema, OpNode, build_op_graph_from_threads
from .scheduler_stub import SchedulePlan, schedule_stub
from .scheduler_toy import schedule_toy_stub
from .topology import Topology, build_topology_from_grid

__all__ = [
    "AbstractEngineConfig",
    "GraphSourceSpec",
    "OpGraph",
    "OpGraphSchema",
    "OpNode",
    "PlannerOptions",
    "SimulatorOptions",
    "ToyGraphGeneratorSpec",
    "Topology",
    "TopologyEdgeSpec",
    "TopologyGridSpec",
    "TopologyGraphSpec",
    "build_op_graph_from_threads",
    "build_topology_from_grid",
    "export_scheduler_input",
    "export_scheduler_input_to_json",
    "load_abstract_engine_config",
    "schedule_stub",
    "schedule_toy_stub",
    "SchedulePlan",
    "SchedulerInput",
    "validate_topology_connectivity",
]
