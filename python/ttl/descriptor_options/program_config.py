# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Program config: ProgramConfig, ProgramRunConfig."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..constants import MemorySpace, Objective, Placement


class ProgramRunConfig(BaseModel):
    """Pydantic config passed to each thread and stored in Program. Replaces injected_program_kwargs dict."""

    grid: list[int] = Field(
        default_factory=lambda: [1, 1], description="Grid dimensions (cols, rows)"
    )
    memory_space: MemorySpace = Field(default=MemorySpace.L1, description="L1 or DRAM")
    tiled: bool = Field(default=True, description="Whether to use tiled layout")
    debug_locations: bool = Field(
        default=True, description="Generate source locations for error messages"
    )

    def inject_into_kwargs(
        self, kwargs: dict[str, object], param_names: set[str]
    ) -> None:
        """Inject only the keys that the kernel function accepts (in-place)."""
        for name in ("grid", "memory_space", "tiled"):
            if name in param_names:
                kwargs[name] = getattr(self, name)


class ProgramConfig(BaseModel):
    """Pydantic model for program_config (grid, objective, placement). Replaces ad-hoc dict handling."""

    grid: tuple[int, int] | None = Field(
        default=None, description="(cols, rows) for scheduler/compile"
    )
    objective: Objective | None = Field(
        default=None, description="Scheduler objective"
    )
    placement: Placement | None = Field(
        default=None, description="Scheduler placement"
    )

    def as_dict(self) -> dict[str, object]:
        """Dict for compatibility with scheduler and CompiledTTNNKernel.program_config."""
        return self.model_dump(exclude_none=False)


__all__ = ["ProgramConfig", "ProgramRunConfig"]
