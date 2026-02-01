# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ...compile.pipeline import _compile_kernel as _compile_kernel_impl
from ...descriptor_options import CompiledTTNNKernel
from ..context import RunContext


def compile_kernel(ctx: RunContext) -> CompiledTTNNKernel | None:
    """Business logic: compile ctx.compile_req and return compiled kernel."""
    if ctx.compile_req is None:
        raise RuntimeError("compile_kernel requires ctx.compile_req to be set")
    return _compile_kernel_impl(ctx.compile_req)


__all__ = ["compile_kernel"]
