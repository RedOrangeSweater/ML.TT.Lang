# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic options for ttnn config descriptors.

Re-exports all public types from submodules. Reduces parameter count and
centralizes validation. See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

from .compile_request import (
    KernelWriteRequest,
    TTNNCompileCacheAndCb,
    TTNNCompileInput,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
    TTNNProfilingInput,
)
from .compile_result import (
    CompiledKernelArtifacts,
    CompiledProfilingSource,
    CompiledRuntimeContext,
    CompiledTTNNKernel,
)
from .core_range import CoreRangeSetOptions
from .program_config import ProgramConfig, ProgramRunConfig
from .thread_config import (
    ComputeConfigOptions,
    NocConfigOptions,
    ReaderConfigOptions,
    ThreadConfigBuildRequest,
)

__all__ = [
    "CompiledKernelArtifacts",
    "CompiledProfilingSource",
    "CompiledRuntimeContext",
    "CompiledTTNNKernel",
    "ComputeConfigOptions",
    "CoreRangeSetOptions",
    "KernelWriteRequest",
    "NocConfigOptions",
    "ProgramConfig",
    "ProgramRunConfig",
    "ReaderConfigOptions",
    "ThreadConfigBuildRequest",
    "TTNNCompileCacheAndCb",
    "TTNNCompileInput",
    "TTNNKernelCompileOptions",
    "TTNNKernelCompileRequest",
    "TTNNProfilingInput",
]
