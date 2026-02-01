# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ...program import ProgramOptions, RunRequest
from ..context import RunContext


def ensure_run_request(ctx: RunContext) -> RunContext:
    """Normalize RunContext raw input into RunRequest (Pydantic-first)."""
    if ctx.req is None:
        if isinstance(ctx.raw_req, RunRequest):
            req = ctx.raw_req
        else:
            if ctx.grid is None:
                raise ValueError(
                    "grid= is required when passing program as first arg; "
                    "e.g. run(add_kernel, lhs, rhs, out, grid=(2, 2))"
                )
            program = ctx.raw_req
            if not callable(program):
                raise TypeError(
                    "run() program must be callable when raw_req is not RunRequest"
                )
            opts = ctx.options if isinstance(ctx.options, ProgramOptions) else None
            req = RunRequest.from_program(
                program, *ctx.raw_args, grid=ctx.grid, options=opts, **ctx.raw_kwargs
            )
        ctx = ctx.model_copy(update={"req": req})
    return ctx


__all__ = ["ensure_run_request"]
