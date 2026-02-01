# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ...compile.pipeline import _compile_kernel as _compile_kernel_impl
from ..context import RunContext


def compile_kernel(ctx: RunContext) -> RunContext:
    """Compile ctx.compile_req and store CompiledTTNNKernel in ctx.compiled."""
    compile_req = ctx.compile_req
    if compile_req is None:
        raise RuntimeError("compile_kernel requires ctx.compile_req to be set")
    compiled = _compile_kernel_impl(compile_req)
    ctx = ctx.model_copy(update={"compiled": compiled})
    return ctx


__all__ = ["compile_kernel"]
