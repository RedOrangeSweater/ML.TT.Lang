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


ensure_run_request_ctx = _wrap_step(ensure_run_request)
resolve_engine_config_ctx = _wrap_step(resolve_engine_config)
build_compile_request_ctx = _wrap_step(build_compile_request)
compile_kernel_ctx = _wrap_step(compile_kernel)


def require_ttl_program_attr_ctx(
    attr_name: str,
) -> Callable[[RunFn[Out_co]], RunFn[Out_co]]:
    """Decorator: enforce TTL program marker on ctx.req.spec.program."""
    return _wrap_step(require_ttl_program_attr(attr_name))


__all__ = [
    "build_compile_request_ctx",
    "compile_kernel_ctx",
    "ensure_run_request_ctx",
    "require_ttl_program_attr_ctx",
    "resolve_engine_config_ctx",
]

