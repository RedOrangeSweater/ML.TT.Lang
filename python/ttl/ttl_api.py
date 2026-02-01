# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Main API for the TTL dialect Python DSL.

Ideal UX (no constraints): ttl.run(add_kernel, lhs, rhs, out, grid=(2,2))
or ttl.run(Spec(program=..., grid=...), *tensors). User thinks only "what to compute"
and "run params"; framework maps to KernelCompileRequest; engine does compile + run.
See docs/sdlc/00_Main/00_Ideas/18_ideal_ux_and_layer_responsibilities.md.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .scheduler import AbstractEngineConfig


try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None


from ._src.auto_profile import is_auto_profile_enabled, run_profiling_after_execute
from .circular_buffer import CircularBuffer
from .compile.compile_thread import compile_thread as _compile_thread_impl
from .constants import MemorySpace
from .descriptor_options import (
    CompiledTTNNKernel,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .program import (
    Program,
    ProgramDecoratorParams,
    ProgramOptions,
    RunRequest,
)
from .settings import settings_ttlang
from .layered import compose
from .layered.context import ProgramInvocationContext, RunContext
from .layered.compile.build_compile_request import build_compile_request
from .layered.compile.compile_kernel import compile_kernel
from .layered.program.ensure_run_request import ensure_run_request
from .layered.program.compute_cache_key import compute_cache_key
from .layered.program.compile_cached import compile_cached
from .layered.program.require_ttl_program import require_ttl_program_attr
from .layered.program.resolve_engine_config import resolve_engine_config


# For kernel body: TensorAccessor and dma (alias for copy) used in examples
# TensorAccessor(tensor) returns the tensor so it is captured; compiler treats
# subscript access (accessor[i,j]) as tensor accessor in MLIR (see ttl_ast).
def TensorAccessor(tensor):
    """Wrap a tensor for use as accessor in DM threads (e.g. accessor[i, j] in copy)."""
    return tensor


dma = copy  # Alias used in examples (DMA = copy for data movement)


def _should_execute() -> bool:
    """Check if kernel execution should proceed (not compile-only mode)."""
    return not settings_ttlang.compile_only


def compute(verbose: bool = False) -> Callable[..., object]:
    """
    Decorator for compute thread functions.

    Compute threads execute on Tensix cores and perform mathematical operations.

    Args:
        verbose: Enable verbose compilation output

    Returns:
        Decorator for compute kernel compilation
    """

    def _decorator(f: Callable[..., object]) -> Callable[..., object]:
        return _compile_thread_impl(f, "compute", verbose)

    return _decorator


def datamovement(verbose: bool = False) -> Callable[..., object]:
    """
    Decorator for data movement thread functions.

    Data movement threads handle DMA operations between memory hierarchies.

    Args:
        verbose: Enable verbose compilation output

    Returns:
        Decorator for data movement kernel compilation
    """

    def _decorator(f: Callable[..., object]) -> Callable[..., object]:
        return _compile_thread_impl(f, "datamovement", verbose)

    return _decorator


# -----------------------------------------------------------------------------
# Compile layer: KernelCompileRequest -> (threads, module) -> TTNNKernelCompileRequest
# -> CompiledTTNNKernel. Implementation in .compile.pipeline; facade delegates.
# -----------------------------------------------------------------------------

OBJECTIVE_VALUES = ("latency", "throughput", "balanced")
PLACEMENT_VALUES = ("auto", "manual")


# -----------------------------------------------------------------------------
# Runtime entry: ProgramSpec + args -> compile -> run_kernel_on_device (kernel_runner).
# Program layer types live in ttl.program; see kernel_runner for Runtime layer.
# -----------------------------------------------------------------------------

# Marker set on @ttl.program-decorated wrappers so run() can accept (program, *args, grid=...).
_TTL_PROGRAM_ATTR = "_ttl_program"


def execute_if_needed(
    compiled: CompiledTTNNKernel | None,
    req: RunRequest,
) -> object | None:
    """Execute compiled kernel if not None and not compile-only mode; else return None."""
    if compiled is None:
        return None
    if _should_execute():
        return compiled(*req.args)
    return None


def execute_and_maybe_profile(
    compiled: CompiledTTNNKernel | None,
    args: tuple[object, ...],
) -> object | None:
    """Execute compiled kernel if not None and not compile-only; run profiling if enabled."""
    if compiled is None or not _should_execute():
        return None
    result = compiled(*args)
    if is_auto_profile_enabled() and compiled.all_source_lines:
        run_profiling_after_execute(
            args,
            compiled.all_source_lines,  # type: ignore[arg-type]
            compiled.thread_to_kernel,
            compiled.kernel_line_offsets,  # type: ignore[arg-type]
        )
    return result


def run(
    req: RunRequest | Callable[..., object],
    *args: object,
    engine_config_path: str | Path | None = None,
    engine_config: AbstractEngineConfig | None = None,
    grid: (tuple[int, ...] | list[int] | Callable[..., object]) | None = None,
    options: ProgramOptions | None = None,
    **kwargs: object,
) -> object | None:
    """
    Compile and run kernel. Ideal UX entry point.

    Two forms:
    - run(RunRequest(spec=spec, args=args, kwargs=kwargs))
    - run(program, *args, grid=..., options=..., **kwargs)
      e.g. run(add_kernel, lhs, rhs, out, grid=(2, 2))

    Optional engine_config_path or engine_config: abstract engine config for scheduler.
    See docs/sdlc/00_Main/00_Ideas/20_nickel_mlir_config_abstract_engine.md.
    """
    ctx = RunContext(
        raw_req=req,
        raw_args=args,
        raw_kwargs=kwargs,
        engine_config_path=engine_config_path,
        engine_config=engine_config,
        grid=grid,
        options=options,
    )

    def handler(ctx: RunContext) -> object | None:
        if ctx.req is None:
            raise RuntimeError("run() pipeline invariant: ctx.req must be set")
        return execute_if_needed(ctx.compiled, ctx.req)

    pipeline = compose(
        handler,
        [
            ensure_run_request,
            resolve_engine_config,
            require_ttl_program_attr(_TTL_PROGRAM_ATTR),
            build_compile_request,
            compile_kernel,
        ],
    )
    return pipeline(ctx)


def pykernel_gen(
    grid: (tuple[int, ...] | Callable[..., object]) | None = None,
    indexing_maps: list[Callable[..., object]] | None = None,
    iterator_types: list[str] | None = None,
    num_outs: int = 1,
    memory_space: MemorySpace = MemorySpace.L1,
    tiled: bool = True,
    fp32_dest_acc_en: bool | None = None,
    dst_full_sync_en: bool | None = None,
    objective: Literal["latency", "throughput", "balanced"] | None = None,
    placement: Literal["auto", "manual"] | None = None,
) -> Callable:
    """
    Decorator for generating TTL kernels from Python functions.

    This decorator compiles Python functions into TTL dialect operations,
    handling thread compilation, stream creation, and pipeline execution.
    Kernels are compiled to C++ for execution via ttnn.generic_op.

    Args:
        grid: Grid dimensions as tuple (e.g., (2, 2)) or callable
        indexing_maps: List of lambda functions for indexing (optional)
        iterator_types: List of iterator types ("parallel", "reduction")
        num_outs: Number of output arguments
        memory_space: MemorySpace (L1 or DRAM)
        tiled: Whether to use tiled layout
        fp32_dest_acc_en: Optional override for fp32_dest_acc_en
        dst_full_sync_en: Optional override for dst_full_sync_en
        objective: Optional policy "latency" | "throughput" | "balanced" (stored, not used yet)
        placement: Optional policy "auto" | "manual" (stored, not used yet)

    Returns:
        Decorated function that compiles and executes the kernel

    Raises:
        AssertionError: If required parameters are missing or invalid
    """
    params = ProgramDecoratorParams(
        grid=grid,
        indexing_maps=indexing_maps,  # type: ignore[arg-type]
        iterator_types=iterator_types,  # type: ignore[arg-type]
        options=ProgramOptions(
            num_outs=num_outs,
            memory_space=memory_space,
            tiled=tiled,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            objective=objective,
            placement=placement,
        ),
    )

    def _decorator(f):
        kernel_id = random.getrandbits(64)
        cache: dict[tuple[object, ...], CompiledTTNNKernel] = {}

        def handler(ctx: ProgramInvocationContext) -> object | None:
            _wrapper._last_compiled_kernel = ctx.compiled  # type: ignore[attr-defined]
            return execute_and_maybe_profile(ctx.compiled, ctx.args)

        pipeline = compose(handler, [compute_cache_key, compile_cached])

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            ctx = ProgramInvocationContext(
                program=f,
                args=args,
                kwargs=kwargs,
                params=params,
                kernel_id=kernel_id,
                cache=cache,
            )
            return pipeline(ctx)

        setattr(_wrapper, _TTL_PROGRAM_ATTR, True)
        return _wrapper

    return _decorator


# Alias for backward compatibility
kernel = pykernel_gen

# Preferred name: one program may compile to one or more device kernels (Reader/Compute/Writer etc.)
program = pykernel_gen


__all__ = [
    "pykernel_gen",
    "kernel",
    "program",
    "Program",
    "compute",
    "datamovement",
    "TensorBlock",
    "CircularBuffer",
    "CopyTransferHandler",
    "copy",
    "dma",
    "TensorAccessor",
    "CompiledTTNNKernel",
]
