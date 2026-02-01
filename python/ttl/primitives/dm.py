# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Data movement primitives: Reader, Writer, DMPrimitive.

Elementary DMA/copy units composable into programs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import BuildContext, Primitive


class DMPrimitive(BaseModel):
    """Base Pydantic model for data movement primitives."""

    name: str = Field(default="", description="Kernel/thread name for diagnostics")

    def build(self, ctx: BuildContext) -> None:
        """Emit DM ops into context. Override in subclasses."""
        pass

    def validate(self) -> None:
        """Validate DM config. Override in subclasses."""
        pass


class Reader(DMPrimitive):
    """Reader (NCRISC) data movement primitive."""

    pass


class Writer(DMPrimitive):
    """Writer (BRISC) data movement primitive."""

    pass


__all__ = ["DMPrimitive", "Reader", "Writer"]
