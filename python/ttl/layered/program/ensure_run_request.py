# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

from ..context import RunContext
from ..core import Handler, Middleware
from ...program import ProgramOptions, RunRequest


def ensure_run_request() -> Middleware[RunContext, object | None]:
    """Normalize RunContext raw input into RunRequest (Pydantic-first)."""

    def mw(next_handler: Handler[RunContext, object | None]) -> Handler[RunContext, object | None]:
        def handler(ctx: RunContext) -> object | None:
            if ctx.req is None:
                if isinstance(ctx.raw_req, RunRequest):
                    req = ctx.raw_req
                else:
                    program = ctx.raw_req
                    if ctx.grid is None:
                        raise ValueError(
                            "grid= is required when passing program as first arg; "
                            "e.g. run(add_kernel, lhs, rhs, out, grid=(2, 2))"
                        )
                    opts = ctx.options if isinstance(ctx.options, ProgramOptions) else None
                    req = RunRequest.from_program(
                        program, *ctx.raw_args, grid=ctx.grid, options=opts, **ctx.raw_kwargs
                    )
                ctx = ctx.model_copy(update={"req": req})
            return next_handler(ctx)

        return handler

    return mw


__all__ = ["ensure_run_request"]

