# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal FastAPI-like middleware composition.

Middleware model:
  - handler(ctx) -> out
  - middleware(next_handler) -> handler
  - compose(handler, [mw1, mw2, ...]) -> wrapped handler

The mental model mirrors MLIR pass pipelines: each middleware is a transform over
an explicit context dialect (Pydantic model), and may update the context or
produce diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


Ctx = TypeVar("Ctx")
Out = TypeVar("Out")

Handler = Callable[[Ctx], Out]
Middleware = Callable[[Handler[Ctx, Out]], Handler[Ctx, Out]]


def compose(handler: Handler[Ctx, Out], middlewares: list[Middleware[Ctx, Out]]) -> Handler[Ctx, Out]:
    """Compose middlewares around a handler (left-to-right order)."""
    wrapped = handler
    for mw in reversed(middlewares):
        wrapped = mw(wrapped)
    return wrapped


__all__ = [
    "Handler",
    "Middleware",
    "compose",
]

