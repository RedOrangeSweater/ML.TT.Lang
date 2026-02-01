# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal FastAPI-like middleware composition.

Middleware model:
  - handler(ctx) -> out
  - middleware(next_handler) -> handler
  - compose(handler, [mw1, mw2, ...]) -> wrapped handler

Business functions use @middleware and take only (ctx) -> ctx; the decorator
calls next_handler(updated_ctx) so the function does not see next_handler.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

Ctx = TypeVar("Ctx")
Out = TypeVar("Out")

Handler = Callable[[Ctx], Out]
Middleware = Callable[[Handler[Ctx, Out]], Handler[Ctx, Out]]


def middleware(fn: Callable[[Ctx], Ctx]) -> Middleware[Ctx, Out]:
    """Wrap a transform (ctx) -> ctx into a Middleware; decorator calls next_handler."""
    @functools.wraps(fn)
    def mw(next_handler: Handler[Ctx, Out]) -> Handler[Ctx, Out]:
        def handler(ctx: Ctx) -> Out:
            return next_handler(fn(ctx))
        return handler
    return mw


def compose(
    handler: Handler[Ctx, Out], middlewares: list[Middleware[Ctx, Out]]
) -> Handler[Ctx, Out]:
    """Compose middlewares around a handler (left-to-right order)."""
    wrapped = handler
    for mw in reversed(middlewares):
        wrapped = mw(wrapped)
    return wrapped


__all__ = [
    "Handler",
    "Middleware",
    "compose",
    "middleware",
]

