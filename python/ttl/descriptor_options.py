# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic options for ttnn config descriptors. Reduces parameter count and centralizes validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ttnn_proxy import (
    ComputeConfigProxy,
    ComputeConfigResolved,
    ComputeDescriptorBuildContext,
    CoreRangeSetProxy,
    ReaderConfigProxy,
    WriterConfigProxy,
)


class ComputeConfigOptions(BaseModel):
    """Options for building ttnn.ComputeConfigDescriptor. Proxy for fp32_dest_acc_en / dst_full_sync_en."""

    fp32_dest_acc_en: bool | None = Field(default=None, description="Enable fp32 destination accumulator")
    dst_full_sync_en: bool | None = Field(default=None, description="Enable destination full sync")

    def build_ttnn_descriptor(self, context: ComputeDescriptorBuildContext) -> tuple[Any, dict[str, str]]:
        """
        Build ttnn.ComputeConfigDescriptor and thread_to_kernel entries.
        When fp32_dest_acc_en is None and context.has_f32 is True, enables fp32_dest_acc_en automatically.
        """
        proxy = ComputeConfigProxy(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
        )
        resolved = ComputeConfigResolved.from_proxy_and_context(proxy, context)
        config = resolved.to_ttnn()
        entries = {
            "TRISC_0": context.kernel_name,
            "TRISC_1": context.kernel_name,
            "TRISC_2": context.kernel_name,
        }
        return config, entries


class NocConfigOptions(BaseModel):
    """Options for building ttnn Reader/Writer config. noc_kernel_idx 0 -> Reader, 1 -> Writer."""

    noc_kernel_idx: int = Field(..., ge=0, le=1, description="0=Reader/NCRISC, 1=Writer/BRISC")
    kernel_name: str = Field(..., description="Kernel name for thread_to_kernel entries")

    def build_ttnn_descriptor(self) -> tuple[Any, dict[str, str]]:
        """Build ttnn.ReaderConfigDescriptor or WriterConfigDescriptor and thread_to_kernel entries."""
        if self.noc_kernel_idx == 0:
            config = ReaderConfigProxy().to_ttnn()
            entries = {"NCRISC": self.kernel_name}
        else:
            config = WriterConfigProxy().to_ttnn()
            entries = {"BRISC": self.kernel_name}
        return config, entries


class ReaderConfigOptions(BaseModel):
    """Options for building ttnn.ReaderConfigDescriptor (fallback for unknown thread_type)."""

    kernel_name: str = Field(default="", description="Kernel name for thread_to_kernel entries")

    def build_ttnn_descriptor(self) -> tuple[Any, dict[str, str]]:
        """Build ttnn.ReaderConfigDescriptor and empty entries (default reader fallback)."""
        config = ReaderConfigProxy().to_ttnn()
        return config, {}


class CoreRangeSetOptions(BaseModel):
    """Options for building ttnn.CoreRangeSet from grid (cols, rows)."""

    grid: tuple[int, int] = Field(
        ...,
        description="(cols, rows) matching tt-metal CoreCoord convention",
    )

    @field_validator("grid")
    @classmethod
    def grid_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        cols, rows = v
        if cols < 1 or rows < 1:
            raise ValueError("grid (cols, rows) must have both >= 1")
        return v

    def build_ttnn_core_range_set(self) -> Any:
        """Build ttnn.CoreRangeSet covering [0,0] to (grid[0]-1, grid[1]-1)."""
        return CoreRangeSetProxy(grid=self.grid).to_ttnn()


class ProgramConfig(BaseModel):
    """Pydantic model for program_config (grid, objective, placement). Replaces ad-hoc dict handling."""

    grid: tuple[int, int] | None = Field(default=None, description="(cols, rows) for scheduler/compile")
    objective: Literal["latency", "throughput", "balanced"] | None = Field(
        default=None, description="Scheduler objective"
    )
    placement: Literal["auto", "manual"] | None = Field(
        default=None, description="Scheduler placement"
    )

    def to_dict(self) -> dict[str, Any]:
        """Dict for compatibility with scheduler and CompiledTTNNKernel.program_config."""
        return self.model_dump(exclude_none=False)


class ThreadConfigBuildRequest(BaseModel):
    """Request for building config for one thread. Replaces 6 parameters in _build_config_for_thread."""

    thread_type: str = Field(..., description="compute | noc | other")
    name: str = Field(..., description="Kernel name")
    noc_kernel_idx: int = Field(default=0, ge=0, description="NOC index (0=Reader, 1=Writer)")
    compute_opts: ComputeConfigOptions = Field(
        default_factory=ComputeConfigOptions,
        description="Compute options when thread_type is compute",
    )
    has_f32: bool = Field(default=False, description="Whether args have float32")
    verbose: bool = Field(default=False, description="Print messages")

    def build_config_and_entries(self) -> tuple[Any, dict[str, str]]:
        """Build (ttnn config descriptor, thread_to_kernel entries) for this thread. Single entry point."""
        if self.thread_type == "compute":
            context = ComputeDescriptorBuildContext(
                kernel_name=self.name,
                has_f32=self.has_f32,
                verbose=self.verbose,
            )
            return self.compute_opts.build_ttnn_descriptor(context)
        if self.thread_type == "noc":
            return NocConfigOptions(
                noc_kernel_idx=self.noc_kernel_idx,
                kernel_name=self.name,
            ).build_ttnn_descriptor()
        return ReaderConfigOptions(kernel_name=self.name).build_ttnn_descriptor()


class TTNNKernelCompileOptions(BaseModel):
    """Single options object for _compile_ttnn_kernel. Replaces many scalar parameters."""

    fp32_dest_acc_en: bool | None = Field(default=None, description="Override for compute fp32 dest acc")
    dst_full_sync_en: bool | None = Field(default=None, description="Override for compute dst full sync")
    verbose: bool = Field(default=True, description="Print compilation info")
    program_config: ProgramConfig | dict[str, Any] | None = Field(
        default=None, description="Grid, objective, placement (dict or ProgramConfig)"
    )

    def compute_config_options(self) -> ComputeConfigOptions:
        """Build ComputeConfigOptions for per-kernel descriptor building."""
        return ComputeConfigOptions(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
        )


class KernelWriteRequest(BaseModel):
    """Request for writing kernel source to a path. Replaces (name, source, base_dir) parameters."""

    name: str = Field(..., description="Kernel name for filename")
    source: str = Field(..., description="Kernel C++ source")
    base_dir: Path | None = Field(default=None, description="Output directory or None for /tmp/{user}")


class TTNNKernelCompileRequest(BaseModel):
    """Single request object for _compile_ttnn_kernel. Replaces 11 positional/keyword parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    module: Any = Field(..., description="MLIR module after D2M pipeline")
    args: Any = Field(..., description="Input/output tensors")
    grid: tuple[int, int] = Field(..., description="(cols, rows)")
    num_outs: int = Field(..., description="Number of output tensors")
    thread_tensor_indices: Any = Field(..., description="Tensor indices per thread")
    cb_configs: Any = Field(default=None, description="Circular buffer configs")
    program_hash: Any = Field(default=None, description="Program cache hash")
    compile_options: TTNNKernelCompileOptions | None = Field(
        default=None, description="fp32/verbose/program_config options"
    )
    source_lines: Any = Field(default=None, description="Source lines for profiling")
    all_source_lines: Any = Field(default=None, description="All thread source lines")
    kernel_line_offsets: Any = Field(default=None, description="Line offsets per kernel")
