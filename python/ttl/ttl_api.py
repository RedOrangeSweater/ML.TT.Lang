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
from collections.abc import Callable, Sequence

from ._src.auto_profile import is_auto_profile_enabled, run_profiling_after_execute
from .boundary.ttnn_types import TtnnTensorLike
from .circular_buffer import CircularBuffer
from .compile.compile_thread import compile_thread as _compile_thread_impl
from .compile.pipeline import _compile_kernel as _compile_kernel_impl
from .compile.registry import get_thread_registry
from .constants import MemorySpace, Objective, Placement
from .descriptor_options import CompiledTTNNKernel
from .layered.context import ProgramInvocationContext, RunContext
from .layered.decorators import (
    cache_by_key,
    ensure_ctx_field,
    require_attr,
    store_to_ctx,
)
from .layered.program.ensure_run_request import ctx_request_ensure_run_request
from .layered.run_decorators import (
    ctx_config_resolve_engine_config,
    ctx_program_require_ttl_program_attr,
    ctx_request_build_run_context,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .program import (
    CompileKernelRequest,
    KernelCompileRequest,
    Program,
    ProgramDecoratorParams,
    ProgramOptions,
    _resolve_grid,
)
from .program.cache_key import make_cache_key
from .settings import settings_ttlang

_LAST_COMPILED_KERNEL_ATTR = "_last_compiled_kernel"


# For kernel body: TensorAccessor and dma (alias for copy) used in examples
# TensorAccessor(tensor) returns the tensor so it is captured; compiler treats
# subscript access (accessor[i,j]) as tensor accessor in MLIR (see ttl_ast).
def TensorAccessor(tensor: TtnnTensorLike) -> TtnnTensorLike:  # noqa: N802
    """Wrap a tensor for use as accessor in DM threads (e.g. accessor[i, j] in copy)."""
    return tensor


dma = copy  # Alias used in examples (DMA = copy for data movement)


def _should_execute() -> bool:
    """Check if kernel execution should proceed (not compile-only mode)."""
    return not settings_ttlang.compile_only


def compute(verbose: bool = False) -> Callable[[Callable[..., object]], Callable[..., object]]:
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


def datamovement(verbose: bool = False) -> Callable[[Callable[..., object]], Callable[..., object]]:
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

# -----------------------------------------------------------------------------
# Runtime entry: ProgramSpec + args -> compile -> run_kernel_on_device (kernel_runner).
# Program layer types live in ttl.program; see kernel_runner for Runtime layer.
# -----------------------------------------------------------------------------

# Marker set on @ttl.program-decorated wrappers so run() can accept
# (program, *args, grid=...).
_TTL_PROGRAM_ATTR = "_ttl_program"


@require_attr("req", error="build_compile_request requires ctx.req to be set")
def build_compile_request(ctx: RunContext) -> CompileKernelRequest:
    """Build CompileKernelRequest from ctx.req and ctx.engine_config."""
    assert ctx.req is not None

    cache_key = make_cache_key(
        ctx.req.args,
        fp32_dest_acc_en=ctx.req.spec.options.fp32_dest_acc_en,
        dst_full_sync_en=ctx.req.spec.options.dst_full_sync_en,
    )
    program_hash = hash((id(ctx.req.spec.program), cache_key))
    grid = _resolve_grid(ctx.req.spec.grid, ctx.req.args, ctx.req.kwargs)
    kernel_req = KernelCompileRequest(
        grid=grid,
        program_hash=program_hash,
        indexing_maps=ctx.req.spec.indexing_maps,
        iterator_types=ctx.req.spec.iterator_types,
        options=ctx.req.spec.options,
    )
    return CompileKernelRequest(
        program=ctx.req.spec.program,
        args=ctx.req.args,
        kwargs=dict(ctx.req.kwargs),
        compile_request=kernel_req,
        thread_registry=get_thread_registry(),
        engine_config=ctx.engine_config,
    )


@require_attr("compile_req", error="compile_kernel requires ctx.compile_req to be set")
def compile_kernel(ctx: RunContext) -> CompiledTTNNKernel | None:
    """Business logic: compile ctx.compile_req and return compiled kernel."""
    assert ctx.compile_req is not None
    return _compile_kernel_impl(ctx.compile_req)


def _compute_program_cache_key(ctx: ProgramInvocationContext) -> tuple[object, ...]:
    return make_cache_key(
        ctx.args,
        fp32_dest_acc_en=ctx.params.options.fp32_dest_acc_en,
        dst_full_sync_en=ctx.params.options.dst_full_sync_en,
    )


@ensure_ctx_field(
    "cache_key",
    _compute_program_cache_key,
    when=lambda ctx: ctx.compiled is None,
)
@store_to_ctx("compiled", skip_if_set=True)
@cache_by_key("cache", "cache_key")
def compile_cached(ctx: ProgramInvocationContext) -> CompiledTTNNKernel | None:
    """Compile with per-kernel cache; wrapping logic is handled by decorators."""
    program_hash = ctx.program_hash
    grid = _resolve_grid(ctx.params.grid, ctx.args, ctx.kwargs)
    kernel_req = KernelCompileRequest(
        grid=grid,
        program_hash=program_hash,
        indexing_maps=ctx.params.indexing_maps,
        iterator_types=ctx.params.iterator_types,
        options=ctx.params.options,
    )
    compile_req = CompileKernelRequest(
        program=ctx.program,
        args=ctx.args,
        kwargs=ctx.kwargs,
        compile_request=kernel_req,
        thread_registry=get_thread_registry(),
        engine_config=None,
    )
    return _compile_kernel_impl(compile_req)


def _pykernel_gen_params_adapter(
    fn: Callable[[ProgramDecoratorParams], Callable],
) -> Callable:
    @functools.wraps(fn, assigned=("__module__", "__name__", "__qualname__"))
    def wrapper(
        grid: tuple[int, ...] | list[int] | Callable[..., object] | str | None = None,
        indexing_maps: Sequence[Callable[..., object]] | None = None,
        iterator_types: Sequence[str] | None = None,
        num_outs: int = 1,
        memory_space: MemorySpace = MemorySpace.L1,
        tiled: bool = True,
        fp32_dest_acc_en: bool | None = None,
        dst_full_sync_en: bool | None = None,
        objective: Objective | None = None,
        placement: Placement | None = None,
    ) -> Callable:
        """Public @ttl.program API: build ProgramDecoratorParams and delegate."""
        params = ProgramDecoratorParams(
            grid=grid,
            indexing_maps=list(indexing_maps) if indexing_maps else [],
            iterator_types=list(iterator_types) if iterator_types else [],
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
        return fn(params)

    return wrapper


@ctx_request_build_run_context
@ctx_request_ensure_run_request
@ctx_config_resolve_engine_config
@ctx_program_require_ttl_program_attr(_TTL_PROGRAM_ATTR)
@ensure_ctx_field("compile_req", build_compile_request)
@ensure_ctx_field("compiled", compile_kernel)
@require_attr("req", error="run() invariant: ctx.req must be set")
def run(ctx: RunContext) -> TtnnTensorLike | None:
    """
    Compile and run kernel. Ideal UX entry point.

    Two forms:
    - run(RunRequest(spec=spec, args=args, kwargs=kwargs))
    - run(program, *args, grid=..., options=..., **kwargs)
      e.g. run(add_kernel, lhs, rhs, out, grid=(2, 2))

    Optional engine_config_path or engine_config: abstract engine config for scheduler.
    See docs/sdlc/00_Main/00_Ideas/20_nickel_mlir_config_abstract_engine.md.
    """
    assert ctx.req is not None
    if ctx.compiled is None:
        return None
    if not _should_execute():
        return None
    result = ctx.compiled(*ctx.req.args)
    return result if isinstance(result, TtnnTensorLike) else None


@_pykernel_gen_params_adapter
def pykernel_gen(params: ProgramDecoratorParams) -> Callable:
    """Decorator entrypoint for pre-built ProgramDecoratorParams."""

    def _decorator(f: Callable) -> Callable:
        kernel_id = random.getrandbits(64)
        cache: dict[tuple[object, ...], CompiledTTNNKernel] = {}

        @functools.wraps(f)
        def _wrapper(*args: TtnnTensorLike, **kwargs: object) -> TtnnTensorLike | None:
            ctx = compile_cached(
                ProgramInvocationContext(
                    program=f,
                    args=args,
                    kwargs=dict(kwargs),
                    params=params,
                    kernel_id=kernel_id,
                    cache=cache,
                )
            )
            setattr(_wrapper, _LAST_COMPILED_KERNEL_ATTR, ctx.compiled)

            if ctx.compiled is None or not _should_execute():
                return None

            result = ctx.compiled(*ctx.args)

            if is_auto_profile_enabled() and ctx.compiled.all_source_lines:
                run_profiling_after_execute(
                    ctx.args,
                    ctx.compiled.all_source_lines,
                    ctx.compiled.thread_to_kernel,
                    ctx.compiled.kernel_line_offsets,
                )
            return result if isinstance(result, TtnnTensorLike) else None  # type: ignore[arg-type]

        setattr(_wrapper, _TTL_PROGRAM_ATTR, True)
        return _wrapper

    return _decorator


def pykernel_from_params(params: ProgramDecoratorParams) -> Callable:
    """Decorator entrypoint for pre-built ProgramDecoratorParams."""
    return pykernel_gen(params)


# Alias for backward compatibility
kernel = pykernel_gen

# Preferred name: one program may compile to one or more device kernels
# (Reader/Compute/Writer etc.).
program = pykernel_gen


__all__ = [
    "pykernel_gen",
    "pykernel_from_params",
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
