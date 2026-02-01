# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import RunContext
from ..core import Handler, Middleware


def resolve_engine_config() -> Middleware[RunContext, object | None]:
    """Resolve engine_config from engine_config_path (if provided)."""

    def mw(next_handler: Handler[RunContext, object | None]) -> Handler[RunContext, object | None]:
        def handler(ctx: RunContext) -> object | None:
            if ctx.engine_config_path is not None:
                from ...scheduler import load_abstract_engine_config

                engine_config = load_abstract_engine_config(ctx.engine_config_path)
                ctx = ctx.model_copy(update={"engine_config": engine_config})
            return next_handler(ctx)

        return handler

    return mw


__all__ = ["resolve_engine_config"]

