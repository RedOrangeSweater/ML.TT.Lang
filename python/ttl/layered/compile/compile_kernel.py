# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import RunContext
from ..core import Handler, Middleware
from ...compile.pipeline import _compile_kernel as _compile_kernel_impl


def compile_kernel() -> Middleware[RunContext, object | None]:
    """Compile ctx.compile_req and store CompiledTTNNKernel in ctx.compiled."""

    def mw(next_handler: Handler[RunContext, object | None]) -> Handler[RunContext, object | None]:
        def handler(ctx: RunContext) -> object | None:
            compile_req = ctx.compile_req
            if compile_req is None:
                raise RuntimeError("compile_kernel requires ctx.compile_req to be set")
            compiled = _compile_kernel_impl(compile_req)
            ctx = ctx.model_copy(update={"compiled": compiled})
            return next_handler(ctx)

        return handler

    return mw


__all__ = ["compile_kernel"]

