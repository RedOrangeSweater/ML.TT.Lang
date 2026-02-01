# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compile.pipeline import _compile_kernel as _compile_kernel_impl
from ...compile.registry import get_thread_registry
from ...program import CompileKernelRequest, KernelCompileRequest, _resolve_grid
from ..context import ProgramInvocationContext
from ..core import middleware

if TYPE_CHECKING:
    from ...descriptor_options import CompiledTTNNKernel


def _compile_program(ctx: ProgramInvocationContext) -> CompiledTTNNKernel | None:
    """Build compile request from ctx and return compiled kernel (no cache)."""
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


@middleware
def compile_cached(ctx: ProgramInvocationContext) -> ProgramInvocationContext:
    """Compile program with per-kernel cache; store compiled kernel in ctx.compiled."""
    if ctx.compiled is None:
        if ctx.cache_key is None:
            raise RuntimeError("compile_cached requires ctx.cache_key to be set")
        if ctx.cache_key in ctx.cache:
            compiled = ctx.cache[ctx.cache_key]
        else:
            compiled = _compile_program(ctx)
            if compiled is not None:
                ctx.cache[ctx.cache_key] = compiled
        ctx = ctx.model_copy(update={"compiled": compiled})
    return ctx


__all__ = ["compile_cached"]
