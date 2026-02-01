# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import ProgramInvocationContext
from ..core import middleware
from ...program.cache_key import make_cache_key


@middleware
def compute_cache_key(ctx: ProgramInvocationContext, next_handler: object) -> object | None:
    """Compute cache_key from args and compile options; store into ctx.cache_key."""
    if ctx.cache_key is None:
        key = make_cache_key(
            ctx.args,
            fp32_dest_acc_en=ctx.params.options.fp32_dest_acc_en,
            dst_full_sync_en=ctx.params.options.dst_full_sync_en,
        )
        ctx = ctx.model_copy(update={"cache_key": key})
    return next_handler(ctx)


__all__ = ["compute_cache_key"]
