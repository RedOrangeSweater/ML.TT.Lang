# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler package (Graph layer): op graph, topology, and scheduler stub.

Graph layer: OpGraph, SchedulePlan, Topology. One focus: operations and
placement plan; R/C/W is an engine placement pattern, not the API axis.
Phase 3-4: build op graph from program threads, topology from grid,
scheduler stub produces plan that reproduces current RCW behavior.
"""

from .op_graph import OpGraph, OpGraphSchema, OpNode, build_op_graph_from_threads
from .scheduler_stub import SchedulePlan, schedule_stub
from .topology import Topology, build_topology_from_grid

from .export import (
    SchedulerInput,
    export_scheduler_input,
    export_scheduler_input_to_json,
)

__all__ = [
    "OpGraph",
    "OpGraphSchema",
    "OpNode",
    "build_op_graph_from_threads",
    "Topology",
    "build_topology_from_grid",
    "schedule_stub",
    "SchedulePlan",
    "SchedulerInput",
    "export_scheduler_input",
    "export_scheduler_input_to_json",
]
