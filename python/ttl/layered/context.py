# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic contexts for layer-level middleware pipelines.

These contexts represent the "dialect" at a layer boundary. Middlewares perform
well-scoped transforms by validating, normalizing, and enriching the context.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..descriptor_options import CompiledTTNNKernel
    from ..program import ProgramDecoratorParams
    from ..program import CompileKernelRequest, ProgramOptions, RunRequest


class RunContext(BaseModel):
    """Context for ttl_api.run() pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Raw input (as given to ttl_api.run)
    raw_req: object = Field(..., description="RunRequest or program callable")
    raw_args: tuple[object, ...] = Field(default_factory=tuple)
    raw_kwargs: dict[str, object] = Field(default_factory=dict)

    # Optional inputs
    engine_config_path: str | Path | None = None
    engine_config: object | None = None
    grid: object | None = None
    options: object | None = None

    # Normalized / derived values
    req: "RunRequest | None" = None
    compile_req: "CompileKernelRequest | None" = None
    compiled: "CompiledTTNNKernel | None" = None


class ProgramInvocationContext(BaseModel):
    """Context for @ttl.program wrapper invocation (pykernel_gen)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: Callable[..., object]
    args: tuple[object, ...] = Field(default_factory=tuple)
    kwargs: dict[str, object] = Field(default_factory=dict)
    params: "ProgramDecoratorParams"
    kernel_id: int
    cache: dict[tuple[object, ...], "CompiledTTNNKernel"] = Field(default_factory=dict)

    cache_key: tuple[object, ...] | None = None
    compiled: "CompiledTTNNKernel | None" = None


__all__ = [
    "RunContext",
    "ProgramInvocationContext",
]

