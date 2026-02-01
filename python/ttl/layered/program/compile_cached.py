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
from ..decorators import cache_by_key, require_attr, store_to_ctx

if TYPE_CHECKING:
    from ...descriptor_options import CompiledTTNNKernel


@middleware
@store_to_ctx("compiled", skip_if_set=True)
@cache_by_key("cache", "cache_key", store_if_not_none=True)
@require_attr("cache_key", error="compile_cached requires ctx.cache_key to be set")
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
