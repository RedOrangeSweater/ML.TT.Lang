# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import RunContext
from ..core import Handler, Middleware
from ...compile.registry import get_thread_registry
from ...program import CompileKernelRequest, KernelCompileRequest, _resolve_grid
from ...program.cache_key import make_cache_key


def build_compile_request() -> Middleware[RunContext, object | None]:
    """Build CompileKernelRequest from RunRequest and store in ctx.compile_req."""

    def mw(next_handler: Handler[RunContext, object | None]) -> Handler[RunContext, object | None]:
        def handler(ctx: RunContext) -> object | None:
            req = ctx.req
            if req is None:
                raise RuntimeError("build_compile_request requires ctx.req to be set")

            cache_key = make_cache_key(
                req.args,
                fp32_dest_acc_en=req.spec.options.fp32_dest_acc_en,
                dst_full_sync_en=req.spec.options.dst_full_sync_en,
            )
            program_hash = hash((id(req.spec.program), cache_key))
            grid = _resolve_grid(req.spec.grid, req.args, req.kwargs)
            kernel_req = KernelCompileRequest(
                grid=grid,
                program_hash=program_hash,
                indexing_maps=req.spec.indexing_maps,
                iterator_types=req.spec.iterator_types,
                options=req.spec.options,
            )
            compile_req = CompileKernelRequest(
                program=req.spec.program,
                args=req.args,
                kwargs=dict(req.kwargs),
                compile_request=kernel_req,
                thread_registry=get_thread_registry(),
                engine_config=ctx.engine_config,
            )
            ctx = ctx.model_copy(update={"compile_req": compile_req})
            return next_handler(ctx)

        return handler

    return mw


__all__ = ["build_compile_request"]

