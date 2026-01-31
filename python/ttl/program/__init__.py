# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Program layer: user intent -> ProgramSpec, KernelCompileRequest.

One focus: grid, options, indexing; no MLIR, no descriptors.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

from typing import Callable, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..constants import MemorySpace
from ..descriptor_options import (
    ProgramConfig,
    ProgramRunConfig,
    TTNNKernelCompileOptions,
)
from ..dtype_utils import is_ttnn_tensor


def _resolve_grid(grid, args, kwargs):
    """Resolve grid, evaluating callable or 'auto' if needed."""
    if callable(grid):
        return grid(*args, **kwargs)
    if grid == "auto":
        for arg in args:
            if is_ttnn_tensor(arg) and hasattr(arg, "device"):
                device = arg.device()
                device_grid = device.compute_with_storage_grid_size()
                return (device_grid.x, device_grid.y)
        raise ValueError(
            "grid='auto' requires at least one ttnn tensor argument "
            "to determine device compute grid"
        )
    return grid


class ProgramDecoratorParams(BaseModel):
    """Validated decorator params for @ttl.program. Contract: grid required; indexing_maps when iterator_types; num_outs == 1."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: (tuple[int, ...] | list[int] | Callable[..., object]) | None = Field(
        default=None,
        description="Grid dimensions or callable (required for TTNN run)",
    )
    indexing_maps: list[Callable[..., object]] | None = Field(
        default=None, description="Indexing maps"
    )
    iterator_types: list[str] | None = Field(default=None, description="Iterator types")
    num_outs: int = Field(1, ge=1)
    memory_space: MemorySpace = Field(default="L1")
    tiled: bool = Field(default=True)
    fp32_dest_acc_en: bool | None = Field(default=None)
    dst_full_sync_en: bool | None = Field(default=None)
    objective: Literal["latency", "throughput", "balanced"] | None = Field(default=None)
    placement: Literal["auto", "manual"] | None = Field(default=None)

    @model_validator(mode="after")
    def grid_required(self) -> Self:
        """Contract: grid parameter is required."""
        if self.grid is None:
            raise ValueError("grid parameter is required")
        return self

    @model_validator(mode="after")
    def indexing_maps_when_iterator_types(self) -> Self:
        """Contract: indexing_maps must be set when iterator_types is set."""
        if self.iterator_types is not None and self.indexing_maps is None:
            raise ValueError("indexing_maps must be set when iterator_types is set")
        return self

    @model_validator(mode="after")
    def single_output_for_run(self) -> Self:
        """Contract: num_outs must be 1 for TTNN run."""
        if self.num_outs != 1:
            raise ValueError(f"num_outs must be 1, got {self.num_outs}")
        return self

    def to_program_options(self) -> "ProgramOptions":
        """Build ProgramOptions from validated params."""
        return ProgramOptions(
            num_outs=self.num_outs,
            memory_space=self.memory_space,
            tiled=self.tiled,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
            objective=self.objective,
            placement=self.placement,
        )


class ProgramOptions(BaseModel):
    """Validated options for @ttl.program decorator. Replaces manual if/raise checks."""

    num_outs: int = Field(1, ge=1)
    memory_space: MemorySpace = "L1"
    tiled: bool = True
    fp32_dest_acc_en: bool | None = None
    dst_full_sync_en: bool | None = None
    objective: Literal["latency", "throughput", "balanced"] | None = None
    placement: Literal["auto", "manual"] | None = None

    def program_config_dict(self) -> dict[str, str | None]:
        """Dict for program_config (objective, placement)."""
        return {"objective": self.objective, "placement": self.placement}

    def to_program_config(self, grid: tuple[int, int] | None = None) -> ProgramConfig:
        """Build ProgramConfig from decorator options (objective, placement, optional grid)."""
        return ProgramConfig(
            grid=grid,
            objective=self.objective,
            placement=self.placement,
        )

    def to_compile_options(
        self,
        *,
        verbose: bool = True,
        program_config: ProgramConfig | dict | None = None,
    ) -> TTNNKernelCompileOptions:
        """Build TTNNKernelCompileOptions for _compile_ttnn_kernel."""
        if program_config is None:
            program_config = self.to_program_config()
        return TTNNKernelCompileOptions(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
            verbose=verbose,
            program_config=program_config,
        )


class KernelCompileRequest(BaseModel):
    """Hierarchical request for _compile_kernel. Per-invocation data at root; decorator options nested in options."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: tuple[int, ...] | list[int] = Field(
        ..., description="Grid dimensions (cols, rows)"
    )
    program_hash: int = Field(..., description="Hash for tt-metal program cache")
    indexing_maps: list[Callable[..., object]] = Field(
        default_factory=list,
        description="Lambda functions for indexing",
    )
    iterator_types: list[str] = Field(
        default_factory=list, description="Iterator type strings"
    )
    options: ProgramOptions = Field(
        ..., description="Decorator-level options (num_outs, memory_space, tiled, etc.)"
    )


class ProgramSpec(BaseModel):
    """Spec for ideal UX: program + grid + options. Used by run(spec, *args)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: Callable[..., object] = Field(
        ..., description="Kernel function (typically @ttl.program-decorated)"
    )
    grid: tuple[int, ...] | list[int] | Callable[..., object] = Field(
        ...,
        description="Grid dimensions or callable to resolve from args",
    )
    options: ProgramOptions = Field(
        ..., description="Program options (memory_space, tiled, etc.)"
    )
    indexing_maps: list[Callable[..., object]] = Field(
        default_factory=list,
        description="Indexing maps for the kernel",
    )
    iterator_types: list[str] = Field(
        default_factory=list, description="Iterator types"
    )

    @model_validator(mode="after")
    def indexing_maps_when_iterator_types(self) -> Self:
        """Contract: indexing_maps must be set when iterator_types is set."""
        if self.iterator_types and not self.indexing_maps:
            raise ValueError("indexing_maps must be set when iterator_types is set")
        return self

    def to_compile_request(
        self,
        args: tuple,
        kwargs: dict,
        program_hash: int,
    ) -> KernelCompileRequest:
        """Build KernelCompileRequest for this spec and invocation."""
        grid = _resolve_grid(self.grid, args, kwargs)
        return KernelCompileRequest(
            grid=grid,
            program_hash=program_hash,
            indexing_maps=self.indexing_maps,
            iterator_types=self.iterator_types,
            options=self.options,
        )


class RunRequest(BaseModel):
    """Request for run(spec, *args): validates num_outs == 1 before compile."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ProgramSpec = Field(
        ..., description="Program spec (program + grid + options)"
    )
    args: tuple[object, ...] = Field(..., description="Positional arguments (tensors)")
    kwargs: dict[str, object] = Field(
        default_factory=dict, description="Keyword arguments"
    )

    @model_validator(mode="after")
    def run_requires_single_output(self) -> Self:
        """Contract: run() supports only num_outs == 1."""
        if self.spec.options.num_outs != 1:
            raise ValueError(f"num_outs must be 1, got {self.spec.options.num_outs}")
        return self


class Program:
    """
    Immutable container for kernel threads and their arguments.

    A Program encapsulates compute and data movement threads along with
    the arguments to be passed during execution. After construction, all
    fields should be treated as read-only.
    """

    def __init__(
        self,
        *threads: object,
        args: tuple[object, ...] = (),
        run_config: ProgramRunConfig | None = None,
        kwargs: dict[str, object] | None = None,
    ):
        self._threads = threads
        self._args = args
        if run_config is not None:
            self._kwargs = run_config.model_dump()
        else:
            self._kwargs = kwargs if kwargs is not None else {}

    @property
    def threads(self) -> tuple[object, ...]:
        return self._threads

    @property
    def args(self) -> tuple[object, ...]:
        return self._args

    @property
    def kwargs(self) -> dict[str, object]:
        return self._kwargs

    def __call__(self, *args: object, **kwargs: object) -> "Program":
        return Program(*self.threads, args=args, kwargs={**self.kwargs, **kwargs})


__all__ = [
    "KernelCompileRequest",
    "Program",
    "ProgramDecoratorParams",
    "ProgramOptions",
    "ProgramSpec",
    "RunRequest",
    "_resolve_grid",
]
