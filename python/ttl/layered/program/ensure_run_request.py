# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import RunContext
from ..core import middleware
from ...program import ProgramOptions, RunRequest


@middleware
def ensure_run_request(ctx: RunContext, next_handler: object) -> object | None:
    """Normalize RunContext raw input into RunRequest (Pydantic-first)."""
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


__all__ = ["ensure_run_request"]
