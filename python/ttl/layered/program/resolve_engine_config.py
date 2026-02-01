# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import RunContext
from ..core import middleware


@middleware
def resolve_engine_config(ctx: RunContext, next_handler: object) -> object | None:
    """Resolve engine_config from engine_config_path (if provided)."""
    if ctx.engine_config_path is not None:
        from ...scheduler import load_abstract_engine_config

        engine_config = load_abstract_engine_config(ctx.engine_config_path)
        ctx = ctx.model_copy(update={"engine_config": engine_config})
    return next_handler(ctx)


__all__ = ["resolve_engine_config"]
