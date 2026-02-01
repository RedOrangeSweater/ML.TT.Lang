# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile layer: KernelCompileRequest -> TTNNKernelCompileRequest -> CompiledTTNNKernel.

Re-exports from descriptor_options. _compile_kernel and _compile_ttnn_kernel in
pipeline; thread_compiler, ttnn_compiler, kernel_writer, source_collector are
internal. One focus: spec to compilation artifacts.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from ..descriptor_options import (
    ComputeConfigOptions,
    CoreRangeSetOptions,
    KernelWriteRequest,
    NocConfigOptions,
    ProgramConfig,
    ProgramRunConfig,
    ReaderConfigOptions,
    ThreadConfigBuildRequest,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
)

__all__ = [
    "ComputeConfigOptions",
    "CoreRangeSetOptions",
    "KernelWriteRequest",
    "NocConfigOptions",
    "ProgramConfig",
    "ProgramRunConfig",
    "ReaderConfigOptions",
    "ThreadConfigBuildRequest",
    "TTNNKernelCompileOptions",
    "TTNNKernelCompileRequest",
]
