# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Layer boundaries for tt-lang Python API.

One focus per layer when reading code. Boundaries use Pydantic request/response.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.

- Program: user intent -> ProgramSpec, KernelCompileRequest (ttl_api).
- Graph: OpGraph, SchedulePlan, Topology (scheduler).
- Compile: KernelCompileRequest -> TTNNKernelCompileRequest -> CompiledTTNNKernel (ttl_api, descriptor_options).
- Runtime: artifacts -> descriptors -> run_kernel_on_device (kernel_runner).
"""

from __future__ import annotations

# Program layer: spec and compile request
from .ttl_api import (
    KernelCompileRequest,
    ProgramOptions,
    ProgramSpec,
    run,
)

# Graph layer: op graph and schedule plan
from .scheduler import (
    OpGraph,
    SchedulePlan,
    Topology,
)

# Compile layer: compile requests and options (descriptor_options)
from .descriptor_options import (
    ThreadConfigBuildRequest,
    TTNNKernelCompileRequest,
    TTNNKernelCompileOptions,
)

# Runtime layer: kernel spec and run request (kernel_runner)
from .kernel_runner import (
    KernelDescriptorBuildRequest,
    CBDescriptorBuildRequest,
    RunKernelRequest,
    KernelSpec,
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
