# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ...compile.pipeline import _compile_kernel as _compile_kernel_impl
from ...compile.registry import get_thread_registry
from ...descriptor_options import CompiledTTNNKernel
from ...program import CompileKernelRequest, KernelCompileRequest, _resolve_grid
from ...program.cache_key import make_cache_key
from ..context import ProgramInvocationContext


def _ensure_cache_key(ctx: ProgramInvocationContext) -> ProgramInvocationContext:
    """Ensure ctx.cache_key is set based on args and compile options."""
    if ctx.cache_key is not None:
        return ctx
    key = make_cache_key(
        ctx.args,
        fp32_dest_acc_en=ctx.params.options.fp32_dest_acc_en,
        dst_full_sync_en=ctx.params.options.dst_full_sync_en,
    )
    return ctx.model_copy(update={"cache_key": key})


def _cache_by_cache_key(fn):
    """Decorator: cache compiled kernel in ctx.cache keyed by ctx.cache_key."""

    def wrapper(ctx: ProgramInvocationContext) -> CompiledTTNNKernel | None:
        ctx = _ensure_cache_key(ctx)
        if ctx.cache_key is None:
            raise RuntimeError("compile_cached requires ctx.cache_key to be set")
        if ctx.cache_key in ctx.cache:
            return ctx.cache[ctx.cache_key]
        compiled = fn(ctx)
        if compiled is not None:
            ctx.cache[ctx.cache_key] = compiled
        return compiled

    return wrapper


def _store_compiled_to_ctx(*, skip_if_set: bool = True):
    """Decorator: write compiled into ctx.compiled; optionally skip if already set."""

    def decorator(fn):
        def wrapper(ctx: ProgramInvocationContext) -> ProgramInvocationContext:
            if skip_if_set and ctx.compiled is not None:
                return ctx
            compiled = fn(ctx)
            return ctx.model_copy(update={"compiled": compiled})

        return wrapper

    return decorator


def _compile_body(ctx: ProgramInvocationContext) -> CompiledTTNNKernel | None:
    """Compile-body only: build requests and return compiled kernel."""
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


_compile_cached_impl = _store_compiled_to_ctx(skip_if_set=True)(
    _cache_by_cache_key(_compile_body)
)


def compile_cached(ctx: ProgramInvocationContext) -> ProgramInvocationContext:
    """Compile with per-kernel cache; wrapping logic is in decorators."""
    return _compile_cached_impl(ctx)


__all__ = ["compile_cached"]
