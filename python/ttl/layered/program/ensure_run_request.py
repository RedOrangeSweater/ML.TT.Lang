# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable

from ...program import ProgramOptions, RunRequest
from ..context import RunContext


def _ensure_run_request(ctx: RunContext) -> RunContext:
    """Normalize RunContext raw input into RunRequest (Pydantic-first)."""
    if ctx.req is None:
        if isinstance(ctx.raw_req, RunRequest):
            req = ctx.raw_req
        else:
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


def ctx_request_ensure_run_request(
    fn: Callable[[RunContext], object | None],
) -> Callable[[RunContext], object | None]:
    """Request-layer decorator: ensure ctx.req is populated before calling fn(ctx)."""

    @functools.wraps(fn)
    def wrapper(ctx: RunContext) -> object | None:
        return fn(_ensure_run_request(ctx))

    return wrapper


__all__ = [
    "ctx_request_ensure_run_request",
]
