# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile result: CompiledTTNNKernel, CompiledKernelArtifacts, CompiledRuntimeContext, CompiledProfilingSource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from ..kernel_runner import RunKernelRequest


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

    def build_run_request(self, *args: object) -> RunKernelRequest:
        """Build RunKernelRequest from this compiled kernel and tensor args (with validation)."""
        from ..kernel_runner import KernelSpec, RunKernelRequest

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

        return RunKernelRequest(
            kernel_specs=kernel_specs,
            tensors=list(args),
            cb_configs=self.cb_configs,
            core_ranges=self.core_ranges,
            program_hash=self.program_hash,
        )

    def __call__(self, *args: object) -> object:
        """Execute the kernel with the given tensors."""
        from ..kernel_runner import run_kernel_on_device

        return run_kernel_on_device(self.build_run_request(*args))

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
        thread_infos = list(
            zip(
                self.thread_names,
                [ty for _, ty in self.kernel_paths],
                strict=False,
            )
        )
        return export_scheduler_input(thread_infos, grid, self.program_config)


__all__ = [
    "CompiledKernelArtifacts",
    "CompiledProfilingSource",
    "CompiledRuntimeContext",
    "CompiledTTNNKernel",
]
