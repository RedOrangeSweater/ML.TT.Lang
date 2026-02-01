# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Core range: CoreRangeSetOptions for building ttnn.CoreRangeSet from grid."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..ttnn_proxy import CoreRangeSetProxy


class CoreRangeSetOptions(BaseModel):
    """Options for building ttnn.CoreRangeSet from grid (cols, rows)."""

    grid: tuple[int, int] = Field(
        ...,
        description="(cols, rows) matching tt-metal CoreCoord convention",
    )

    @field_validator("grid")
    @classmethod
    def grid_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        cols, rows = v
        if cols < 1 or rows < 1:
            raise ValueError("grid (cols, rows) must have both >= 1")
        return v

    def build_ttnn_core_range_set(self) -> object:
        """Build ttnn.CoreRangeSet covering [0,0] to (grid[0]-1, grid[1]-1)."""
        return CoreRangeSetProxy(grid=self.grid).build_ttnn()


__all__ = ["CoreRangeSetOptions"]
