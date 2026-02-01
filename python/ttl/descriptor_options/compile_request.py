# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile request: TTNNKernelCompileRequest, TTNNCompileInput, TTNNKernelCompileOptions, KernelWriteRequest, etc."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..boundary import MlirModuleLike
    validate_kernel_count_for_request,
    validate_ttnn_tensors_for_request,
)

from .program_config import ProgramConfig
from .thread_config import ComputeConfigOptions


class KernelWriteRequest(BaseModel):
    """Request for writing kernel source to a path. Replaces (name, source, base_dir) parameters."""

    name: str = Field(..., description="Kernel name for filename")
    source: str = Field(..., description="Kernel C++ source")
    base_dir: Path | None = Field(
        default=None, description="Output directory or None for /tmp/{user}"
    )


class TTNNCompileInput(BaseModel):
    """Input to TTNN kernel compilation: module, tensors, grid, num_outs, thread tensor indices."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    module: MlirModuleLike = Field(..., description="MLIR module after D2M pipeline")
    args: tuple[object, ...] = Field(..., description="Input/output tensors")
    grid: tuple[int, int] = Field(..., description="(cols, rows)")
    num_outs: int = Field(..., description="Number of output tensors")
    thread_tensor_indices: list[list[int]] = Field(
        ..., description="Tensor indices per thread"
    )


class TTNNCompileCacheAndCb(BaseModel):
    """Cache and circular buffer configs for TTNN compile request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cb_configs: list[object] | None = Field(
        default=None, description="Circular buffer configs"
    )
    program_hash: int | None = Field(default=None, description="Program cache hash")


class TTNNProfilingInput(BaseModel):
    """Profiling/debug input: source lines and kernel line offsets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_lines: list[str] | None = Field(
        default=None, description="Source lines for profiling"
    )
    all_source_lines: dict[str, list[str]] | None = Field(
        default=None, description="All thread source lines"
    )
    kernel_line_offsets: dict[str, int] | None = Field(
        default=None, description="Line offsets per kernel"
    )


class TTNNKernelCompileOptions(BaseModel):
    """Single options object for _compile_ttnn_kernel. Replaces many scalar parameters."""

    fp32_dest_acc_en: bool | None = Field(
        default=None, description="Override for compute fp32 dest acc"
    )
    dst_full_sync_en: bool | None = Field(
        default=None, description="Override for compute dst full sync"
    )
    verbose: bool = Field(default=True, description="Print compilation info")
    program_config: ProgramConfig | None = Field(
        default=None, description="Grid, objective, placement"
    )

    def compute_config_options(self) -> ComputeConfigOptions:
        """Build ComputeConfigOptions for per-kernel descriptor building."""
        return ComputeConfigOptions(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
        )


class TTNNKernelCompileRequest(BaseModel):
    """Single request object for _compile_ttnn_kernel. Structured as input, options, cache_and_cb, profiling."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: TTNNCompileInput = Field(
        ...,
        description="Compilation input (module, args, grid, num_outs, thread_tensor_indices)",
    )
    compile_options: TTNNKernelCompileOptions | None = Field(
        default=None, description="fp32/verbose/program_config options"
    )
    cache_and_cb: TTNNCompileCacheAndCb | None = Field(
        default=None, description="CB configs and program cache hash"
    )
    profiling: TTNNProfilingInput | None = Field(
        default=None, description="Source lines for profiling"
    )

    @model_validator(mode="after")
    def validate_ttnn_interop(self) -> Self:
        """TTNN interop: all tensors same type (TTNN), L1/DRAM, interleaved, tilized; exactly 3 kernels."""
        from ..compile.validation import (
            validate_kernel_count_for_request,
            validate_ttnn_tensors_for_request,
        )

        args = self.input.args
        validate_ttnn_tensors_for_request(
            args if isinstance(args, tuple) else tuple()
        )
        validate_kernel_count_for_request(self.input.module)
        return self


__all__ = [
    "KernelWriteRequest",
    "TTNNCompileCacheAndCb",
    "TTNNCompileInput",
    "TTNNKernelCompileOptions",
    "TTNNKernelCompileRequest",
    "TTNNProfilingInput",
]
