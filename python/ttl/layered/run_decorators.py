# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable

from .context import RunContext
from .program.require_ttl_program import require_ttl_program_attr
from .program.resolve_engine_config import resolve_engine_config

RunFn = Callable[[RunContext], object | None]


def _wrap_step(
    step: Callable[[RunContext], RunContext],
) -> Callable[[RunFn], RunFn]:
    """Turn a ctx->ctx step into a decorator for a ctx->Out business fn."""

    def decorator(fn: RunFn) -> RunFn:
        @functools.wraps(fn)
        def wrapper(ctx: RunContext) -> object | None:
            return fn(step(ctx))

        return wrapper

    return decorator


def ctx_config_resolve_engine_config(fn: RunFn) -> RunFn:
    """Decorator: resolve ctx.engine_config from ctx.engine_config_path."""
    return _wrap_step(resolve_engine_config)(fn)


def ctx_program_require_ttl_program_attr(
    attr_name: str,
) -> Callable[[RunFn], RunFn]:
    """Decorator: enforce TTL program marker on ctx.req.spec.program."""
    return _wrap_step(require_ttl_program_attr(attr_name))


def ctx_request_build_run_context(
    fn: RunFn,
) -> Callable[..., object | None]:
    """
    Decorator: build RunContext from run() args/kwargs and call fn(ctx).

    This is the outermost decorator for the public ttl_api.run entrypoint.
    """

    @functools.wraps(fn)
    def wrapper(
        req: object,
        *args: object,
        engine_config_path: object | None = None,
        engine_config: object | None = None,
        grid: object | None = None,
        options: object | None = None,
        **kwargs: object,
    ) -> object | None:
        ctx = RunContext(
            raw_req=req,  # type: ignore[arg-type]
            raw_args=args,
            raw_kwargs=kwargs,
            engine_config_path=engine_config_path,  # type: ignore[arg-type]
            engine_config=engine_config,
            grid=grid,  # type: ignore[arg-type]
            options=options,  # type: ignore[arg-type]
        )
        return fn(ctx)

    return wrapper


__all__ = [
    "ctx_config_resolve_engine_config",
    "ctx_program_require_ttl_program_attr",
    "ctx_request_build_run_context",
]

