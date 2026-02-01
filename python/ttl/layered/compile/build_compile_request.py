# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ...compile.registry import get_thread_registry
from ...program import CompileKernelRequest, KernelCompileRequest, _resolve_grid
from ...program.cache_key import make_cache_key
from ..context import RunContext


def build_compile_request(ctx: RunContext) -> CompileKernelRequest:
    """Business logic: build CompileKernelRequest from ctx.req and ctx.engine_config."""
    if ctx.req is None:
        raise RuntimeError("build_compile_request requires ctx.req to be set")

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


__all__ = ["build_compile_request"]
