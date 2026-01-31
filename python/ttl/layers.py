# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Layer boundaries for tt-lang Python API.

One focus per layer when reading code. Boundaries use Pydantic request/response.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.

- Program: user intent -> ProgramSpec, KernelCompileRequest (ttl.program).
- Graph: OpGraph, SchedulePlan, Topology (ttl.graph -> scheduler).
- Compile: KernelCompileRequest -> TTNNKernelCompileRequest -> CompiledTTNNKernel (ttl_api, ttl.compile).
- Runtime: artifacts -> descriptors -> run_kernel_on_device (ttl.runtime -> kernel_runner).
"""

from __future__ import annotations

# Program layer: spec and compile request
from .program import (
    KernelCompileRequest,
    ProgramOptions,
    ProgramSpec,
)
from .ttl_api import run

# Graph layer: op graph and schedule plan
from .graph import (
    OpGraph,
    SchedulePlan,
    Topology,
)

# Compile layer: compile requests and options
from .compile import (
    ThreadConfigBuildRequest,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
)

# Runtime layer: kernel spec and run request
from .runtime import (
    CBDescriptorBuildRequest,
    KernelDescriptorBuildRequest,
    KernelSpec,
    RunKernelRequest,
)

__all__ = [
    # Program
    "KernelCompileRequest",
    "ProgramOptions",
    "ProgramSpec",
    "run",
    # Graph
    "OpGraph",
    "SchedulePlan",
    "Topology",
    # Compile
    "ThreadConfigBuildRequest",
    "TTNNKernelCompileOptions",
    "TTNNKernelCompileRequest",
    # Runtime
    "CBDescriptorBuildRequest",
    "KernelDescriptorBuildRequest",
    "KernelSpec",
    "RunKernelRequest",
]
