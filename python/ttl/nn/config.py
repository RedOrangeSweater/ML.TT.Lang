# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic config models for ttl.nn pipeline, sequential, and Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

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


class TrainerConfig(BaseModel):
    """Config for ttl.nn.Trainer: epochs, steps, compile warmup, profiling, device, output_dir.

    Used for benchmark/inference loop and scheduler-training loop. Validation in Pydantic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_epochs: int = Field(default=1, ge=0, description="Max epochs (0 = run once)")
    max_steps: int | None = Field(
        default=None,
        ge=0,
        description="Max steps per run (None = use max_epochs only)",
    )
    compile_warmup_steps: int = Field(
        default=1,
        ge=0,
        description="Steps to run before timing (compile cache warmup)",
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable TTLANG_AUTO_PROFILE-style profiling; artifacts in output_dir",
    )
    device_id: int = Field(default=0, ge=0, description="Device id for ttnn.open_device (when device used)")
    output_dir: Path | str = Field(
        default=".",
        description="Directory for metrics jsonl, profiles, checkpoints",
    )
    mode: Literal["benchmark", "fit", "validate", "predict", "scheduler_env"] = Field(
        default="benchmark",
        description="Trainer mode: benchmark (inference timing), fit/validate/predict, or scheduler_env (reward loop)",
    )
