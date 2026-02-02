# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

Ctx = TypeVar("Ctx")
Val = TypeVar("Val")
Out = TypeVar("Out")
P = ParamSpec("P")
R = TypeVar("R")


def require_attr(attr_name: str, *, error: str | None = None):
    """Require ctx.<attr_name> to be non-None before calling the function."""

    def decorator(fn: Callable[[Ctx], Val]) -> Callable[[Ctx], Val]:
        @functools.wraps(fn)
        def wrapper(ctx: Ctx) -> Val:
            if getattr(ctx, attr_name) is None:
                raise RuntimeError(error or f"required attribute is None: {attr_name}")
            return fn(ctx)

        return wrapper

    return decorator


def cache_by_key(
    cache_attr: str,
    key_attr: str,
    *,
    store_if_not_none: bool = True,
):
    """
    Cache decorator: lookup getattr(ctx, cache_attr)[getattr(ctx, key_attr)].

    - On hit: returns cached value (fn is not called).
    - On miss: calls fn(ctx); stores the result if enabled.
    """

    def decorator(fn: Callable[[Ctx], Val]) -> Callable[[Ctx], Val]:
        @functools.wraps(fn)
        def wrapper(ctx: Ctx) -> Val:
            cache = getattr(ctx, cache_attr)
            key = getattr(ctx, key_attr)

            if key is not None and key in cache:
                return cache[key]

            value = fn(ctx)
            if store_if_not_none and value is not None and key is not None:
                cache[key] = value
            return value

        return wrapper

    return decorator


def store_to_ctx(field: str, *, skip_if_set: bool = True):
    """Convert fn(ctx)->value into fn(ctx)->ctx via ctx.model_copy()."""

    def decorator(fn: Callable[[Ctx], Any]) -> Callable[[Ctx], Ctx]:
        @functools.wraps(fn)
        def wrapper(ctx: Ctx) -> Ctx:
            if skip_if_set and getattr(ctx, field) is not None:
                return ctx
            value = fn(ctx)
            model_copy = getattr(ctx, "model_copy", None)
            if model_copy is None:
                raise TypeError("store_to_ctx requires ctx.model_copy(...)")
            return model_copy(update={field: value})

        return wrapper

    return decorator


def ensure_ctx_field(
    field: str,
    compute: Callable[[Ctx], Any],
    *,
    skip_if_set: bool = True,
    when: Callable[[Ctx], bool] | None = None,
):
    """
    Ensure ctx.<field> is set before calling the function.

    This is a common pattern for Pydantic contexts:
    - if ctx.<field> is None: ctx = ctx.model_copy(update={field: compute(ctx)})
    - then: return fn(ctx)
    """

    def decorator(fn: Callable[[Ctx], Out]) -> Callable[[Ctx], Out]:
        @functools.wraps(fn)
        def wrapper(ctx: Ctx) -> Out:
            if when is not None and not when(ctx):
                return fn(ctx)
            if skip_if_set and getattr(ctx, field) is not None:
                return fn(ctx)
            model_copy = getattr(ctx, "model_copy", None)
            if model_copy is None:
                raise TypeError("ensure_ctx_field requires ctx.model_copy(...)")
            ctx = model_copy(update={field: compute(ctx)})
            return fn(ctx)

        return wrapper

    return decorator


def require_module_available(get_module: Callable[[], object | None], *, name: str):
    """Require an optional dependency module to be available before calling the function."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if get_module() is None:
                raise RuntimeError(f"{name} is not available")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "cache_by_key",
    "ensure_ctx_field",
    "require_module_available",
    "require_attr",
    "store_to_ctx",
]

