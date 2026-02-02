# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile request: TTNNKernelCompileRequest, TTNNCompileInput, TTNNKernelCompileOptions, KernelWriteRequest, etc."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..boundary import MlirModuleLike
from ..constants import SUPPORTED_MEMORY_SPACES, MemorySpace
from ..dtype_utils import TTNNMemoryConfigProxy, is_ttnn_tensor
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

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        """Validate tensor types and kernel count."""
        # 1. Validate tensors
        ttnn_count = sum(1 for arg in self.args if is_ttnn_tensor(arg))
        if ttnn_count > 0 and ttnn_count < len(self.args):
            raise ValueError(
                f"TTNN interop requires all tensors to be the same type. "
                f"Got {ttnn_count} TTNN tensors and {len(self.args) - ttnn_count} host tensors."
            )
        for i, arg in enumerate(self.args):
            if not is_ttnn_tensor(arg):
                continue
            proxy = TTNNMemoryConfigProxy(tensor=arg, default=MemorySpace.UNKNOWN)
            if proxy.memory_space not in SUPPORTED_MEMORY_SPACES:
                raise ValueError(
                    f"TTNN interop requires L1 or DRAM memory space, but tensor {i} is in {proxy.memory_space}."
                )
            if not proxy.is_interleaved:
                raise ValueError(
                    f"TTNN interop requires interleaved tensors, but tensor {i} is not."
                )
            if hasattr(arg, "layout") and "TILE" not in str(arg.layout):
                raise ValueError(
                    f"TTNN interop requires tilized tensors, but tensor {i} has layout {arg.layout}."
                )

        # 2. Validate kernel count
        from ttmlir.passes import get_ttkernel_names

        kernel_info = get_ttkernel_names(self.module)
        if len(kernel_info) != 3:
            compute_count = sum(1 for _, t in kernel_info if t == "compute")
            dm_count = sum(1 for _, t in kernel_info if t == "noc")
            raise ValueError(
                f"TTNN interop requires exactly 3 kernels (1 compute + 2 data movement), "
                f"got {len(kernel_info)} kernels ({compute_count} compute, {dm_count} data movement)."
            )
        return self


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
        # Logic moved to TTNNCompileInput.validate_input
        return self


__all__ = [
    "KernelWriteRequest",
    "TTNNCompileCacheAndCb",
    "TTNNCompileInput",
    "TTNNKernelCompileOptions",
    "TTNNKernelCompileRequest",
    "TTNNProfilingInput",
]
