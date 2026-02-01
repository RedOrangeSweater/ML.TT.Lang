# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Protocol, TypeVar

from .compile.build_compile_request import build_compile_request
from .compile.compile_kernel import compile_kernel
from .context import RunContext
from .program.ensure_run_request import ensure_run_request
from .program.require_ttl_program import require_ttl_program_attr
from .program.resolve_engine_config import resolve_engine_config

Out_co = TypeVar("Out_co", covariant=True)


class RunFn(Protocol[Out_co]):
    def __call__(self, ctx: RunContext) -> Out_co: ...


def _wrap_step(
    step: Callable[[RunContext], RunContext],
) -> Callable[[RunFn[Out_co]], RunFn[Out_co]]:
    """Turn a ctx->ctx step into a decorator for a ctx->Out business fn."""

    def decorator(fn: RunFn[Out_co]) -> RunFn[Out_co]:
        @functools.wraps(fn)
        def wrapper(ctx: RunContext) -> Out_co:
            return fn(step(ctx))

        return wrapper

    return decorator


ctx_request_ensure_run_request = _wrap_step(ensure_run_request)
ctx_config_resolve_engine_config = _wrap_step(resolve_engine_config)
ctx_compile_build_compile_request = _wrap_step(build_compile_request)
ctx_compile_compile_kernel = _wrap_step(compile_kernel)


def ctx_program_require_ttl_program_attr(
    attr_name: str,
) -> Callable[[RunFn[Out_co]], RunFn[Out_co]]:
    """Decorator: enforce TTL program marker on ctx.req.spec.program."""
    return _wrap_step(require_ttl_program_attr(attr_name))


def ctx_request_build_run_context(
    fn: Callable[[RunContext], object | None],
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
    "ctx_compile_build_compile_request",
    "ctx_compile_compile_kernel",
    "ctx_config_resolve_engine_config",
    "ctx_program_require_ttl_program_attr",
    "ctx_request_build_run_context",
    "ctx_request_ensure_run_request",
]

