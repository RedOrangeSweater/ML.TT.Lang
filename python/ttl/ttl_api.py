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
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from .scheduler import AbstractEngineConfig


try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None


from ._src.auto_profile import is_auto_profile_enabled, run_profiling_after_execute
from .circular_buffer import CircularBuffer
from .compile.compile_thread import compile_thread as _compile_thread_impl
from .compile.pipeline import _compile_kernel as _compile_kernel_impl
from .compile.registry import get_thread_registry
from .constants import MemorySpace
from .descriptor_options import (
    CompiledTTNNKernel,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .program import (
    CompileKernelRequest,
    KernelCompileRequest,
    Program,
    ProgramDecoratorParams,
    ProgramOptions,
    RunRequest,
    _resolve_grid,
)
from .program.cache_key import make_cache_key
from .settings import settings_ttlang


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
    if not isinstance(req, RunRequest):
        program = req
        if grid is None:
            raise ValueError(
                "grid= is required when passing program as first arg; "
                "e.g. run(add_kernel, lhs, rhs, out, grid=(2, 2))"
            )
        req = RunRequest.from_program(
            program, *args, grid=grid, options=options, **kwargs
        )

    if engine_config_path is not None:
        from .scheduler import load_abstract_engine_config

        engine_config = load_abstract_engine_config(engine_config_path)
    if not getattr(req.spec.program, _TTL_PROGRAM_ATTR, False):
        raise NotImplementedError(
            "run() with a raw callable (e.g. lambda) is not yet implemented: "
            "inference from parameters and return type is planned. "
            "Use @ttl.program to define the kernel and pass RunRequest.from_program(program, *args, grid=...). "
            "See docs/sdlc/00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md "
            "(lambda + inference)."
        )
    cache_key = make_cache_key(
        req.args,
        fp32_dest_acc_en=req.spec.options.fp32_dest_acc_en,
        dst_full_sync_en=req.spec.options.dst_full_sync_en,
    )
    program_hash = hash((id(req.spec.program), cache_key))
    compile_request = req.spec.build_compile_request(req.args, req.kwargs, program_hash)
    compile_req = CompileKernelRequest(
        program=req.spec.program,
        args=req.args,
        kwargs=dict(req.kwargs),
        compile_request=compile_request,
        thread_registry=get_thread_registry(),
        engine_config=engine_config,
    )
    compiled = cast(
        CompiledTTNNKernel | None,
        _compile_kernel_impl(compile_req),
    )
    if compiled is None:
        return None
    if _should_execute():
        return compiled(*req.args)
    return None


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
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
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
    options = params.options
    indexing_maps = params.indexing_maps if params.indexing_maps is not None else []
    iterator_types = params.iterator_types if params.iterator_types is not None else []

    def _decorator(f):
        # Per-kernel state: random ID and cache
        kernel_id = random.getrandbits(64)
        cache: dict[tuple[object, ...], CompiledTTNNKernel] = {}

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            resolved_grid = _resolve_grid(grid, args, kwargs)

            # Build cache key from tensor properties
            cache_key = make_cache_key(
                args,
                fp32_dest_acc_en=options.fp32_dest_acc_en,
                dst_full_sync_en=options.dst_full_sync_en,
            )

            # Check cache for previously compiled kernel
            if cache_key in cache:
                compiled_kernel = cache[cache_key]
            else:
                # Compute program_hash for tt-metal cache
                program_hash = hash((kernel_id, cache_key))

                # Compile kernel
                compile_request = KernelCompileRequest(
                    grid=cast(tuple[int, ...] | list[int], resolved_grid),
                    program_hash=program_hash,
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types,
                    options=options,
                )
                compile_req = CompileKernelRequest(
                    program=f,
                    args=args,
                    kwargs=kwargs,
                    compile_request=compile_request,
                    thread_registry=get_thread_registry(),
                    engine_config=None,
                )
                raw_compiled = _compile_kernel_impl(compile_req)
                compiled_kernel = cast(CompiledTTNNKernel | None, raw_compiled)

                if compiled_kernel is not None:
                    cache[cache_key] = compiled_kernel

            # Expose last compiled kernel for scheduler export (e.g. get_scheduler_input)
            _wrapper._last_compiled_kernel = compiled_kernel

            # Execute (unless compile-only mode)
            if compiled_kernel is not None and _should_execute():
                result = compiled_kernel(*args)

                # Run auto-profiling after execution
                if is_auto_profile_enabled() and compiled_kernel.all_source_lines:
                    run_profiling_after_execute(
                        args,
                        compiled_kernel.all_source_lines,  # type: ignore[arg-type]
                        compiled_kernel.thread_to_kernel,
                        compiled_kernel.kernel_line_offsets,  # type: ignore[arg-type]
                    )

                return result

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
