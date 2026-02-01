# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import ProgramInvocationContext
from ..core import Handler, Middleware
from ...program.cache_key import make_cache_key


def compute_cache_key() -> Middleware[ProgramInvocationContext, object | None]:
    """Compute cache_key from args and compile options; store into ctx.cache_key."""

    def mw(
        next_handler: Handler[ProgramInvocationContext, object | None],
    ) -> Handler[ProgramInvocationContext, object | None]:
        def handler(ctx: ProgramInvocationContext) -> object | None:
            if ctx.cache_key is None:
                key = make_cache_key(
                    ctx.args,
                    fp32_dest_acc_en=ctx.params.options.fp32_dest_acc_en,
                    dst_full_sync_en=ctx.params.options.dst_full_sync_en,
                )
                ctx = ctx.model_copy(update={"cache_key": key})
            return next_handler(ctx)

        return handler

    return mw


__all__ = ["compute_cache_key"]

