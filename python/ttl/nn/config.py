# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic config models for ttl.nn pipeline and sequential composition."""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from ..program import ProgramOptions


class PipelineConfig(BaseModel):
    """Config for pipeline/sequential run: default grid and program options.

    Validation is in Pydantic; used when building Module from program + grid.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: tuple[int, ...] | list[int] | Callable[..., object] | None = Field(
        default=None,
        description="Default grid dimensions or callable (required per-step if not set)",
    )
    options: ProgramOptions = Field(
        default_factory=lambda: ProgramOptions(),
        description="Default program options (memory_space, tiled, etc.)",
    )
