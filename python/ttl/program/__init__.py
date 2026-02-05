# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Program layer: user intent -> ProgramSpec, KernelCompileRequest.

One focus: grid, options, indexing; no MLIR, no descriptors.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from ..boundary.ttnn_types import TtnnTensorLike
from ..constants import MemorySpace, Objective, Placement
from ..descriptor_options import (
    ProgramConfig,
    ProgramRunConfig,
    TTNNKernelCompileOptions,
)
from ..dtype_utils import is_ttnn_tensor
from ..scheduler import AbstractEngineConfig

PROGRAM_ATTR = "_ttl_program"
PROGRAM_DECORATOR_PARAMS_ATTR = "_ttl_program_params"
PROGRAM_HASH_ATTR = "_ttl_program_hash"


def _as_grid_2d(grid: object) -> tuple[int, int]:
    """Normalize grid input to a 2D (cols, rows) tuple of ints."""
    if not isinstance(grid, (tuple, list)):
        raise TypeError(
            f"grid must be tuple/list/callable/'auto', got {type(grid).__name__}"
        )
    if len(grid) != 2:
        raise ValueError(f"Only 2D grids supported, got grid {tuple(grid)}")
    return (int(grid[0]), int(grid[1]))


Grid2D = Annotated[tuple[int, int], BeforeValidator(_as_grid_2d)]


class ThreadRegistryLike(Protocol):
    """Protocol for thread registry used by compile pipeline."""

    def clear(self) -> None: ...
    def get_and_clear(self) -> list[Callable[..., object]]: ...


def _resolve_grid(
    grid: (tuple[int, ...] | list[int] | Callable[..., object]) | str | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[int, int]:
    """Resolve grid (callable/'auto'), returning a concrete 2D tuple."""
    if callable(grid):
        resolved = grid(*args, **kwargs)
        if not isinstance(resolved, (tuple, list)):
            raise TypeError(
                f"grid callable must return tuple/list, got {type(resolved).__name__}"
            )
        return _as_grid_2d(resolved)
    if grid == "auto":
        for arg in args:
            if is_ttnn_tensor(arg) and hasattr(arg, "device"):
                # Use TtnnTensorLike protocol for type safety
                tensor: TtnnTensorLike = arg  # type: ignore[assignment]
                device = tensor.device()
                device_grid = device.compute_with_storage_grid_size()
                return (device_grid.x, device_grid.y)
        raise ValueError(
            "grid='auto' requires at least one ttnn tensor argument "
            "to determine device compute grid"
        )
    if not isinstance(grid, (tuple, list)):
        raise TypeError(
            f"grid must be tuple/list/callable/'auto', got {type(grid).__name__}"
        )
    return _as_grid_2d(grid)


def _none_to_list(v: object) -> list[object]:
    """Pydantic BeforeValidator: normalize None to empty list."""
    return [] if v is None else v  # type: ignore[return-value]


IndexingMaps = Annotated[list[Callable[..., object]], BeforeValidator(_none_to_list)]
IteratorTypes = Annotated[list[str], BeforeValidator(_none_to_list)]


class ProgramOptions(BaseModel):
    """Validated options for @ttl.program decorator. Replaces manual if/raise checks."""

    num_outs: int = Field(default=1, ge=1)
    memory_space: MemorySpace = MemorySpace.L1
    tiled: bool = True
    fp32_dest_acc_en: bool | None = None
    dst_full_sync_en: bool | None = None
    objective: Objective | None = None
    placement: Placement | None = None

    @property
    def program_config_dict(self) -> dict[str, str | None]:
        """Dict for program_config (objective, placement)."""
        return {
            "objective": self.objective.value if self.objective else None,
            "placement": self.placement.value if self.placement else None,
        }

    def build_program_config(
        self,
        grid: tuple[int, int] | None = None,
    ) -> ProgramConfig:
        """Build ProgramConfig from decorator options and optional grid."""
        return ProgramConfig(
            grid=grid,
            objective=self.objective,
            placement=self.placement,
        )

    def build_compile_options(
        self,
        *,
        verbose: bool = True,
        program_config: ProgramConfig | None = None,
    ) -> TTNNKernelCompileOptions:
        """Build TTNNKernelCompileOptions for _compile_ttnn_kernel."""
        if program_config is None:
            program_config = self.build_program_config()
        return TTNNKernelCompileOptions(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
            verbose=verbose,
            program_config=program_config,
        )


class ProgramDecoratorParams(BaseModel):
    """Validated decorator params for @ttl.program.

    Contract: grid required; indexing_maps when iterator_types; options.num_outs == 1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: (tuple[int, ...] | list[int] | Callable[..., object] | str) | None = Field(
        default=None,
        description="Grid dimensions or callable (required for TTNN run)",
    )
    indexing_maps: IndexingMaps = Field(
        default_factory=list, description="Indexing maps (None normalized to [])"
    )
    iterator_types: IteratorTypes = Field(
        default_factory=list, description="Iterator types (None normalized to [])"
    )
    options: ProgramOptions = Field(
        default_factory=lambda: ProgramOptions(),
        description="Program options (num_outs, memory_space, tiled, etc.)",
    )

    @model_validator(mode="after")
    def grid_required(self) -> Self:
        """Contract: grid parameter is required."""
        if self.grid is None:
            raise ValueError("grid parameter is required")
        return self

    @model_validator(mode="after")
    def grid_string_value(self) -> Self:
        """Contract: string grid values must be 'auto'."""
        if isinstance(self.grid, str) and self.grid != "auto":
            raise ValueError("grid string must be 'auto'")
        return self

    @model_validator(mode="after")
    def indexing_maps_when_iterator_types(self) -> Self:
        """Contract: indexing_maps must be set when iterator_types is set."""
        if self.iterator_types and not self.indexing_maps:
            raise ValueError("indexing_maps must be set when iterator_types is set")
        return self

    @model_validator(mode="after")
    def single_output_for_run(self) -> Self:
        """Contract: num_outs must be 1 for TTNN run."""
        if self.options.num_outs != 1:
            raise ValueError(f"num_outs must be 1, got {self.options.num_outs}")
        return self

    @model_validator(mode="after")
    def indexing_maps_dims_match_iterator_types(self) -> Self:
        """Contract: indexing_map params count matches len(iterator_types)."""
        if not self.indexing_maps or not self.iterator_types:
            return self
        it_len = len(self.iterator_types)
        for i, indexing_map in enumerate(self.indexing_maps):
            try:
                sig = inspect.signature(indexing_map)
                num_dims = len(list(sig.parameters))
            except (TypeError, ValueError):
                continue
            if num_dims != it_len:
                raise ValueError(
                    f"Number of dimensions ({num_dims}) must match iterator_types "
                    f"length ({it_len}) for indexing_map[{i}]"
                )
        return self


class KernelCompileRequest(BaseModel):
    """Hierarchical request for _compile_kernel.

    Per-invocation data at root; decorator options nested in options.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Grid2D = Field(
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


class CompileKernelRequest(BaseModel):
    """Single request object for _compile_kernel.

    Bundles program, invocation, compile params, and registry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: Callable[..., object] = Field(
        ..., description="Kernel function (typically @ttl.program-decorated)"
    )
    args: tuple[object, ...] = Field(
        default_factory=tuple, description="Positional arguments (tensors)"
    )
    init_kwargs: dict[str, object] = Field(
        default_factory=dict, description="Keyword arguments", alias="kwargs"
    )
    compile_request: KernelCompileRequest = Field(
        ..., description="Grid, program_hash, options for this compile"
    )
    thread_registry: ThreadRegistryLike = Field(
        ..., description="Registry for @compute/@datamovement threads"
    )
    engine_config: AbstractEngineConfig | None = Field(
        default=None, description="Optional abstract engine config for scheduler"
    )

    @property
    def kwargs(self) -> dict[str, object]:
        """Alias for init_kwargs for call sites expecting .kwargs."""
        return self.init_kwargs


class ProgramSpec(BaseModel):
    """Spec for ideal UX: program + grid + options. Used by run(spec, *args)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: Callable[..., object] = Field(
        ..., description="Kernel function (typically @ttl.program-decorated)"
    )
    grid: tuple[int, ...] | list[int] | Callable[..., object] | str = Field(
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

    @model_validator(mode="after")
    def grid_string_value(self) -> Self:
        """Contract: string grid values must be 'auto'."""
        if isinstance(self.grid, str) and self.grid != "auto":
            raise ValueError("grid string must be 'auto'")
        return self

class RunRequest(BaseModel):
    """Request for run(req).

    Single entry point: spec + args + kwargs. Validates num_outs == 1 before compile.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ProgramSpec = Field(
        ..., description="Program spec (program + grid + options)"
    )
    args: tuple[object, ...] = Field(..., description="Positional arguments (tensors)")
    init_kwargs: dict[str, object] = Field(
        default_factory=dict, description="Keyword arguments", alias="kwargs"
    )

    @model_validator(mode="after")
    def run_requires_single_output(self) -> Self:
        """Contract: run() supports only num_outs == 1."""
        if self.spec.options.num_outs != 1:
            raise ValueError(f"num_outs must be 1, got {self.spec.options.num_outs}")
        return self

    @property
    def kwargs(self) -> dict[str, object]:
        """Alias for init_kwargs for call sites expecting .kwargs."""
        return self.init_kwargs

    @classmethod
    def from_program(
        cls,
        program: Callable[..., object],
        *args: object,
        grid: (tuple[int, ...] | list[int] | Callable[..., object] | str) | None = None,
        options: ProgramOptions | None = None,
        **kwargs: object,
    ) -> Self:
        """Build RunRequest from a @ttl.program-decorated callable.

        If grid is omitted, decorator params are used when available.
        """
        params = getattr(program, PROGRAM_DECORATOR_PARAMS_ATTR, None)
        if params is not None and not isinstance(params, ProgramDecoratorParams):
            params = None
        if grid is None:
            if isinstance(params, ProgramDecoratorParams):
                grid = params.grid
            else:
                raise ValueError(
                    "grid= is required when passing program without decorator params; "
                    "e.g. RunRequest.from_program(add_kernel, lhs, rhs, out, grid=(2, 2))"
                )
        opts = options if options is not None else (
            params.options if isinstance(params, ProgramDecoratorParams) else ProgramOptions()
        )
        indexing_maps = (
            list(params.indexing_maps)
            if isinstance(params, ProgramDecoratorParams)
            else []
        )
        iterator_types = (
            list(params.iterator_types)
            if isinstance(params, ProgramDecoratorParams)
            else []
        )
        spec = ProgramSpec(
            program=program,
            grid=grid,
            options=opts,
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
        )
        return cls(spec=spec, args=args, kwargs=kwargs)


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

    def __call__(self, *args: object, **kwargs: object) -> Program:
        return Program(*self.threads, args=args, kwargs={**self.kwargs, **kwargs})


from .builder import ProgramBuilder  # noqa: E402

__all__ = [
    "CompileKernelRequest",
    "KernelCompileRequest",
    "Program",
    "ProgramBuilder",
    "ProgramDecoratorParams",
    "ProgramOptions",
    "ProgramSpec",
    "RunRequest",
    "_resolve_grid",
]
