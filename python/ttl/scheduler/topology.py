# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Topology model: grid of cores for scheduler (Phase 4).

Minimal model: rectangular grid, nodes = cores (coordinates).
No per-core resource detail for stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Topology:
    """Rectangular grid of cores (cols x rows)."""

    grid_cols: int
    grid_rows: int

    def num_cores(self) -> int:
        return self.grid_cols * self.grid_rows

    def core_coords(self) -> list[tuple[int, int]]:
        """Yield (col, row) for each core in row-major order."""
        return [(c, r) for r in range(self.grid_rows) for c in range(self.grid_cols)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for export (e.g. JSON for scheduler viz)."""
        return {"grid_cols": self.grid_cols, "grid_rows": self.grid_rows}


def build_topology_from_grid(grid: tuple[int, ...] | list[int]) -> Topology:
    """Build topology from grid dimensions (cols, rows)."""
    if len(grid) < 2:
        raise ValueError(f"Grid must have at least 2 dimensions, got {grid}")
    return Topology(grid_cols=grid[0], grid_rows=grid[1])
