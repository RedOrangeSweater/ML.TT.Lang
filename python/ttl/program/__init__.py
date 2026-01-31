# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Program layer: user intent -> ProgramSpec, KernelCompileRequest.

One focus: grid, options, indexing; no MLIR, no descriptors.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..descriptor_options import (
    ProgramConfig,
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


class ProgramOptions(BaseModel):
    """Validated options for @ttl.program decorator. Replaces manual if/raise checks."""

    num_outs: int = Field(1, ge=1)
    memory_space: Literal["L1", "DRAM"] = "L1"
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

    grid: tuple[int, ...] | list[int] = Field(..., description="Grid dimensions (cols, rows)")
    program_hash: int = Field(..., description="Hash for tt-metal program cache")
    indexing_maps: list[Callable[..., object]] = Field(
        default_factory=list,
        description="Lambda functions for indexing",
    )
    iterator_types: list[str] = Field(default_factory=list, description="Iterator type strings")
    options: ProgramOptions = Field(..., description="Decorator-level options (num_outs, memory_space, tiled, etc.)")


class ProgramSpec(BaseModel):
    """Spec for ideal UX: program + grid + options. Used by run(spec, *args)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: Callable[..., object] = Field(..., description="Kernel function (typically @ttl.program-decorated)")
    grid: tuple[int, ...] | list[int] | Callable[..., object] = Field(
        ...,
        description="Grid dimensions or callable to resolve from args",
    )
    options: ProgramOptions = Field(..., description="Program options (memory_space, tiled, etc.)")
    indexing_maps: list[Callable[..., object]] = Field(
        default_factory=list,
        description="Indexing maps for the kernel",
    )
    iterator_types: list[str] = Field(default_factory=list, description="Iterator types")

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


__all__ = [
    "_resolve_grid",
    "KernelCompileRequest",
    "ProgramOptions",
    "ProgramSpec",
]
