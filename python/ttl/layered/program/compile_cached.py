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
from ..decorators import cache_by_key, ensure_ctx_field, store_to_ctx


def _compute_cache_key(ctx: ProgramInvocationContext) -> tuple[object, ...]:
    return make_cache_key(
        ctx.args,
        fp32_dest_acc_en=ctx.params.options.fp32_dest_acc_en,
        dst_full_sync_en=ctx.params.options.dst_full_sync_en,
    )


@ensure_ctx_field(
    "cache_key",
    _compute_cache_key,
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


__all__ = ["compile_cached"]
