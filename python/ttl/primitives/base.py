# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Base primitives: Protocol and composable primitive for PyTorch-like modular assembly.

Primitives are elementary DM, compute, or sync units that can be composed into programs.
See docs/sdlc/00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class BuildContext(BaseModel):
    """Context passed to primitive.build() for MLIR/descriptor construction."""

    model_config = {"arbitrary_types_allowed": True}

    grid: tuple[int, ...] = Field(default=(1, 1), description="Grid dimensions")
    memory_space: str = Field(default="L1", description="L1 or DRAM")


@runtime_checkable
class Primitive(Protocol):
    """Base protocol for composable primitives (DM, compute, sync)."""

    def build(self, ctx: BuildContext) -> None:
        """Emit MLIR or update descriptor state. Called during compilation."""
        ...

    def validate(self) -> None:
        """Validate primitive configuration. Raises on error."""
        ...


class ComposablePrimitive(BaseModel):
    """Pydantic-based composable primitive: list of child primitives."""

    model_config = {"arbitrary_types_allowed": True}

    children: list[object] = Field(
        default_factory=list,
        description="Child primitives (DM, compute, sync) in execution order",
    )

    def compose(self, *primitives: Primitive) -> ComposablePrimitive:
        """Return a new ComposablePrimitive with additional children."""
        return self.model_copy(
            update={"children": [*self.children, *primitives]}
        )


__all__ = ["BuildContext", "ComposablePrimitive", "Primitive"]
