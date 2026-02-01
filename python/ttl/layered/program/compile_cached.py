# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ...compile.pipeline import _compile_kernel as _compile_kernel_impl
from ...compile.registry import get_thread_registry
from ...program import CompileKernelRequest, KernelCompileRequest, _resolve_grid
from ..context import ProgramInvocationContext
from ..core import middleware

if TYPE_CHECKING:
    from ...descriptor_options import CompiledTTNNKernel


def _requires_cache_key(
    fn: Callable[[ProgramInvocationContext], object],
) -> Callable[[ProgramInvocationContext], object]:
    """Decorator: raise if ctx.cache_key is None before calling fn."""

    def wrapper(ctx: ProgramInvocationContext) -> object:
        if ctx.cache_key is None:
            raise RuntimeError("compile_cached requires ctx.cache_key to be set")
        return fn(ctx)

    return wrapper


def _cached_by_cache_key(
    fn: Callable[[ProgramInvocationContext], object],
) -> Callable[[ProgramInvocationContext], object]:
    """Lookup ctx.cache by cache_key; on miss call fn and store if not None."""

    def wrapper(ctx: ProgramInvocationContext) -> object:
        if ctx.cache_key is not None and ctx.cache_key in ctx.cache:
            return ctx.cache[ctx.cache_key]
        compiled = fn(ctx)
        if compiled is not None and ctx.cache_key is not None:
            ctx.cache[ctx.cache_key] = compiled  # type: ignore[assignment]
        return compiled

    return wrapper


def _store_compiled_to_ctx(
    fn: Callable[[ProgramInvocationContext], object],
) -> Callable[[ProgramInvocationContext], ProgramInvocationContext]:
    """If ctx.compiled is set return ctx; else call fn, set ctx.compiled, return ctx."""

    def wrapper(ctx: ProgramInvocationContext) -> ProgramInvocationContext:
        if ctx.compiled is not None:
            return ctx
        compiled = fn(ctx)
        return ctx.model_copy(update={"compiled": compiled})

    return wrapper


@middleware
@_store_compiled_to_ctx
@_cached_by_cache_key
@_requires_cache_key
def compile_cached(ctx: ProgramInvocationContext) -> CompiledTTNNKernel | None:
    """Compile program (body only); cache and ctx updates are in decorators."""
    program_hash = hash((ctx.kernel_id, ctx.cache_key))
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


__all__ = ["compile_cached"]
