# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compute primitives: ComputePrimitive, ElementwiseOp, MatmulOp.

Elementary compute units (elementwise, matmul) composable into programs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import BuildContext, Primitive


class ComputePrimitive(BaseModel):
    """Base Pydantic model for compute primitives."""

    name: str = Field(default="", description="Kernel/thread name for diagnostics")

    def build(self, ctx: BuildContext) -> None:
        """Emit compute ops into context. Override in subclasses."""
        pass

    def validate(self) -> None:
        """Validate compute config. Override in subclasses."""
        pass


class ElementwiseOp(ComputePrimitive):
    """Elementwise operation primitive (e.g. add, mul)."""

    op_name: str = Field(default="add", description="Operation name for IR")


class MatmulOp(ComputePrimitive):
    """Matrix multiply primitive."""

    tile_m: int = Field(default=32, ge=1, description="Tile rows")
    tile_n: int = Field(default=32, ge=1, description="Tile cols")
    tile_k: int = Field(default=32, ge=1, description="Tile inner dim")


__all__ = ["ComputePrimitive", "ElementwiseOp", "MatmulOp"]
