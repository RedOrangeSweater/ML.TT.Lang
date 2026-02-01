# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline stages: features enabled when included in the pipeline.

Each stage has a name, enabled(settings), and run(context). The compile pipeline
runs only stages that are in the registry and enabled. State is passed via
CompileStageContext (Pydantic).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class CompileStageContext(BaseModel):
    """State passed between pipeline stages. Stages read/write as needed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    module: Any = Field(default=None, description="MLIR module at current pipeline point")
    initial_mlir_path: str | None = Field(
        default=None, description="Path to save initial MLIR when stage enabled"
    )
    final_mlir_path: str | None = Field(
        default=None, description="Path to save final MLIR when stage enabled"
    )
    print_debug_locations: bool = Field(
        default=False, description="Include debug info when printing MLIR"
    )


@runtime_checkable
class Stage(Protocol):
    """Protocol for a pipeline stage: name, enabled predicate, and run."""

    @property
    def name(self) -> str:
        """Stage name for logging and registry."""
        ...

    def enabled(self, settings: Any) -> bool:
        """Return True if this stage should run given settings."""
        ...

    def run(self, context: CompileStageContext) -> None:
        """Run the stage. Read/write context as needed."""
        ...


def run_stages(
    stages: list[Stage],
    context: CompileStageContext,
    settings: Any,
) -> None:
    """Run each stage that is enabled. Order is preserved."""
    for stage in stages:
        if stage.enabled(settings):
            stage.run(context)


def get_default_stages() -> list[Stage]:
    """Return the default list of stages for the compile pipeline."""
    from .save_initial_mlir import SaveInitialMlirStage

    return [SaveInitialMlirStage()]


def get_initial_stages() -> list[Stage]:
    """Stages run at the start of the compile pipeline (e.g. save initial MLIR)."""
    from .save_initial_mlir import SaveInitialMlirStage

    return [SaveInitialMlirStage()]


def get_final_stages() -> list[Stage]:
    """Stages run at the end of the compile pipeline (e.g. save final MLIR)."""
    from .save_final_mlir import SaveFinalMlirStage

    return [SaveFinalMlirStage()]


__all__ = [
    "CompileStageContext",
    "Stage",
    "get_default_stages",
    "get_final_stages",
    "get_initial_stages",
    "run_stages",
]
