# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic options for ttnn config descriptors. Reduces parameter count and centralizes validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .boundary import MlirModuleLike
from .constants import MemorySpace
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

    fp32_dest_acc_en: bool | None = Field(
        default=None, description="Enable fp32 destination accumulator"
    )
    dst_full_sync_en: bool | None = Field(
        default=None, description="Enable destination full sync"
    )

    def build_ttnn_descriptor(
        self, context: ComputeDescriptorBuildContext
    ) -> tuple[object, dict[str, str]]:
        """
        Build ttnn.ComputeConfigDescriptor and thread_to_kernel entries.
        When fp32_dest_acc_en is None and context.has_f32 is True, enables fp32_dest_acc_en automatically.
        """
        proxy = ComputeConfigProxy(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
        )
        resolved = ComputeConfigResolved.from_proxy_and_context(proxy, context)
        config = resolved.build_ttnn()
        entries = {
            "TRISC_0": context.kernel_name,
            "TRISC_1": context.kernel_name,
            "TRISC_2": context.kernel_name,
        }
        return config, entries


class NocConfigOptions(BaseModel):
    """Options for building ttnn Reader/Writer config. noc_kernel_idx 0 -> Reader, 1 -> Writer."""

    noc_kernel_idx: int = Field(
        ..., ge=0, le=1, description="0=Reader/NCRISC, 1=Writer/BRISC"
    )
    kernel_name: str = Field(
        ..., description="Kernel name for thread_to_kernel entries"
    )

    def build_ttnn_descriptor(self) -> tuple[object, dict[str, str]]:
        """Build ttnn.ReaderConfigDescriptor or WriterConfigDescriptor and thread_to_kernel entries."""
        if self.noc_kernel_idx == 0:
            config = ReaderConfigProxy().build_ttnn()
            entries = {"NCRISC": self.kernel_name}
        else:
            config = WriterConfigProxy().build_ttnn()
            entries = {"BRISC": self.kernel_name}
        return config, entries


class ReaderConfigOptions(BaseModel):
    """Options for building ttnn.ReaderConfigDescriptor (fallback for unknown thread_type)."""

    kernel_name: str = Field(
        default="", description="Kernel name for thread_to_kernel entries"
    )

    def build_ttnn_descriptor(self) -> tuple[object, dict[str, str]]:
        """Build ttnn.ReaderConfigDescriptor and empty entries (default reader fallback)."""
        config = ReaderConfigProxy().build_ttnn()
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

    def build_ttnn_core_range_set(self) -> object:
        """Build ttnn.CoreRangeSet covering [0,0] to (grid[0]-1, grid[1]-1)."""
        return CoreRangeSetProxy(grid=self.grid).build_ttnn()


class ProgramRunConfig(BaseModel):
    """Pydantic config passed to each thread and stored in Program. Replaces injected_program_kwargs dict."""

    grid: list[int] = Field(
        default_factory=lambda: [1, 1], description="Grid dimensions (cols, rows)"
    )
    memory_space: MemorySpace = Field(default=MemorySpace.L1, description="L1 or DRAM")
    tiled: bool = Field(default=True, description="Whether to use tiled layout")
    debug_locations: bool = Field(
        default=True, description="Generate source locations for error messages"
    )

    def inject_into_kwargs(
        self, kwargs: dict[str, object], param_names: set[str]
    ) -> None:
        """Inject only the keys that the kernel function accepts (in-place)."""
        for name in ("grid", "memory_space", "tiled"):
            if name in param_names:
                kwargs[name] = getattr(self, name)


class ProgramConfig(BaseModel):
    """Pydantic model for program_config (grid, objective, placement). Replaces ad-hoc dict handling."""

    grid: tuple[int, int] | None = Field(
        default=None, description="(cols, rows) for scheduler/compile"
    )
    objective: Literal["latency", "throughput", "balanced"] | None = Field(
        default=None, description="Scheduler objective"
    )
    placement: Literal["auto", "manual"] | None = Field(
        default=None, description="Scheduler placement"
    )

    def as_dict(self) -> dict[str, object]:
        """Dict for compatibility with scheduler and CompiledTTNNKernel.program_config."""
        return self.model_dump(exclude_none=False)


class ThreadConfigBuildRequest(BaseModel):
    """Request for building config for one thread. Replaces 6 parameters in _build_config_for_thread."""

    thread_type: str = Field(..., description="compute | noc | other")
    name: str = Field(..., description="Kernel name")
    noc_kernel_idx: int = Field(
        default=0, ge=0, description="NOC index (0=Reader, 1=Writer)"
    )
    compute_opts: ComputeConfigOptions = Field(
        default_factory=ComputeConfigOptions,
        description="Compute options when thread_type is compute",
    )
    has_f32: bool = Field(default=False, description="Whether args have float32")
    verbose: bool = Field(default=False, description="Print messages")

    def build_config_and_entries(self) -> tuple[object, dict[str, str]]:
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


class CompiledKernelArtifacts(BaseModel):
    """Output of compilation per kernel: paths, configs, arg specs, tensor indices, thread mapping."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kernel_paths: list[tuple[str, str]] = Field(
        ...,
        description="List of (path, thread_type) tuples for each kernel",
    )
    kernel_configs: list[object] = Field(
        ...,
        description="List of config descriptors matching kernel_paths",
    )
    kernel_arg_specs: list[object] = Field(
        ...,
        description="List of arg specs (rt_args list) for each kernel",
    )
    kernel_tensor_indices: list[list[int]] = Field(
        ...,
        description="List of global tensor indices used by each kernel",
    )
    thread_to_kernel: dict[str, str] = Field(
        default_factory=dict,
        description="Dict mapping RISC thread name to kernel name",
    )
    thread_names: list[str] = Field(
        default_factory=list,
        description="Thread names in same order as kernel_paths (for scheduler export)",
    )

    @field_validator("thread_names", mode="before")
    @classmethod
    def _none_to_list(cls, v: object) -> object:
        return v if v is not None else []

    @field_validator("thread_to_kernel", mode="before")
    @classmethod
    def _none_to_dict(cls, v: object) -> object:
        return v if v is not None else {}


class CompiledRuntimeContext(BaseModel):
    """What runtime needs to execute: tensors count, core ranges, CB configs, program hash, program config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_tensors: int = Field(..., ge=0, description="Number of input/output tensors")
    core_ranges: object = Field(..., description="CoreRangeSet for kernel execution")
    cb_configs: list[object] = Field(
        default_factory=list,
        description="CircularBuffer configs per CB index",
    )
    program_hash: object | None = Field(
        default=None, description="Hash for tt-metal program cache"
    )
    program_config: dict[str, object] = Field(
        default_factory=dict,
        description="Grid, objective, placement, etc.",
    )

    @field_validator("cb_configs", mode="before")
    @classmethod
    def _cb_none_to_list(cls, v: object) -> object:
        return v if v is not None else []

    @field_validator("program_config", mode="before")
    @classmethod
    def _program_config_none_to_dict(cls, v: object) -> object:
        return v if v is not None else {}


class CompiledProfilingSource(BaseModel):
    """Source lines for profiling and debugging (deprecated source_lines, all_source_lines, kernel_line_offsets)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_lines: object | None = Field(
        default=None, description="Source lines (deprecated)"
    )
    all_source_lines: dict[str, object] = Field(
        default_factory=dict,
        description="Dict mapping kernel name to source lines",
    )
    kernel_line_offsets: dict[str, object] = Field(
        default_factory=dict,
        description="Dict mapping kernel name to line offset",
    )

    @field_validator("all_source_lines", "kernel_line_offsets", mode="before")
    @classmethod
    def _none_to_dict(cls, v: object) -> object:
        return v if v is not None else {}


class CompiledTTNNKernel(BaseModel):
    """
    A compiled tt-lang kernel ready for execution via ttnn.generic_op.

    Caches compilation artifacts (kernel paths, CB descriptors) so the kernel
    can be executed multiple times with different tensors without recompiling.
    Structured as artifacts (per-kernel output), runtime (execution context),
    and optional profiling (source lines).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts: CompiledKernelArtifacts = Field(
        ...,
        description="Per-kernel compilation output (paths, configs, thread mapping)",
    )
    runtime: CompiledRuntimeContext = Field(
        ...,
        description="Execution context (num_tensors, core_ranges, cb_configs, etc.)",
    )
    profiling: CompiledProfilingSource | None = Field(
        default=None, description="Source lines for profiling/debugging"
    )

    @property
    def kernel_paths(self) -> list[tuple[str, str]]:
        return self.artifacts.kernel_paths

    @property
    def kernel_configs(self) -> list[object]:
        return self.artifacts.kernel_configs

    @property
    def kernel_arg_specs(self) -> list[object]:
        return self.artifacts.kernel_arg_specs

    @property
    def kernel_tensor_indices(self) -> list[list[int]]:
        return self.artifacts.kernel_tensor_indices

    @property
    def thread_to_kernel(self) -> dict[str, str]:
        return self.artifacts.thread_to_kernel

    @property
    def thread_names(self) -> list[str]:
        return self.artifacts.thread_names

    @property
    def num_tensors(self) -> int:
        return self.runtime.num_tensors

    @property
    def core_ranges(self) -> object:
        return self.runtime.core_ranges

    @property
    def cb_configs(self) -> list[object]:
        return self.runtime.cb_configs

    @property
    def program_hash(self) -> object | None:
        return self.runtime.program_hash

    @property
    def program_config(self) -> dict[str, object]:
        return self.runtime.program_config

    @property
    def source_lines(self) -> object | None:
        return self.profiling.source_lines if self.profiling else None

    @property
    def all_source_lines(self) -> dict[str, object]:
        return self.profiling.all_source_lines if self.profiling else {}

    @property
    def kernel_line_offsets(self) -> dict[str, object]:
        return self.profiling.kernel_line_offsets if self.profiling else {}

    def __call__(self, *args: object) -> object:
        """Execute the kernel with the given tensors."""
        from .kernel_runner import (
            KernelSpec,
            RunKernelRequest,
            run_kernel_on_device,
        )

        if len(args) != self.num_tensors:
            raise ValueError(f"Expected {self.num_tensors} tensors, got {len(args)}")

        device = args[0].device()
        device_grid = device.compute_with_storage_grid_size()
        kernel_grid = self.core_ranges.bounding_box().grid_size()
        if kernel_grid.x > device_grid.x or kernel_grid.y > device_grid.y:
            raise ValueError(
                f"Kernel grid ({kernel_grid.x}, {kernel_grid.y}) exceeds device "
                f"compute grid ({device_grid.x}, {device_grid.y}). "
                f"Reduce grid size to fit within available cores."
            )

        kernel_specs = []
        for kernel_idx, (kernel_path, thread_type) in enumerate(self.kernel_paths):
            tensor_indices = self.kernel_tensor_indices[kernel_idx]
            config = self.kernel_configs[kernel_idx]
            spec = KernelSpec(
                path=kernel_path,
                thread_type=thread_type,
                tensor_indices=tensor_indices,
                config=config,
            )
            kernel_specs.append(spec)

        run_req = RunKernelRequest(
            kernel_specs=kernel_specs,
            tensors=list(args),
            cb_configs=self.cb_configs,
            core_ranges=self.core_ranges,
            program_hash=self.program_hash,
        )
        return run_kernel_on_device(run_req)

    def get_scheduler_input(self) -> dict:
        """Return scheduler input (op_graph, topology, plan) for use by scheduler-viz."""
        from ..scheduler import export_scheduler_input

        if not self.thread_names or len(self.thread_names) != len(self.kernel_paths):
            raise ValueError(
                "get_scheduler_input requires thread_names (same length as kernel_paths). "
                "Recompile the kernel to get scheduler export support."
            )
        grid = self.program_config.get("grid")
        if grid is None and self.core_ranges is not None:
            box = self.core_ranges.bounding_box()
            g = box.grid_size()
            grid = (g.x, g.y)
        if grid is None:
            raise ValueError(
                "Cannot derive grid for scheduler input (no program_config.grid or core_ranges)."
            )
        thread_infos = list(zip(self.thread_names, [ty for _, ty in self.kernel_paths], strict=False))
        return export_scheduler_input(thread_infos, grid, self.program_config)


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
        from .compile.validation import (
            validate_kernel_count_for_request,
            validate_ttnn_tensors_for_request,
        )

        args = self.input.args
        validate_ttnn_tensors_for_request(args if isinstance(args, tuple) else tuple())
        validate_kernel_count_for_request(self.input.module)
        return self
