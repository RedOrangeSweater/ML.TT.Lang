# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ProgramBuilder: fluent builder for ProgramSpec.

Example:
    spec = (ProgramBuilder(add_kernel)
        .with_grid((2, 2))
        .with_options(memory_space=MemorySpace.L1)
        .build())
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from ..constants import MemorySpace
from . import ProgramOptions, ProgramSpec


class ProgramBuilder:
    """Fluent builder for ProgramSpec."""

    def __init__(self, program: Callable[..., object]) -> None:
        self._program = program
        self._grid: tuple[int, ...] | list[int] | Callable[..., object] = (1, 1)
        self._options = ProgramOptions()
        self._indexing_maps: list[Callable[..., object]] = []
        self._iterator_types: list[str] = []

    def with_grid(
        self,
        grid: tuple[int, ...] | list[int] | Callable[..., object],
    ) -> Self:
        """Set grid dimensions or callable."""
        self._grid = grid
        return self

    def with_options(
        self,
        *,
        num_outs: int | None = None,
        memory_space: MemorySpace | None = None,
        tiled: bool | None = None,
        fp32_dest_acc_en: bool | None = None,
        dst_full_sync_en: bool | None = None,
        objective: str | None = None,
        placement: str | None = None,
    ) -> Self:
        """Set program options. Keyword args override defaults."""
        opts = self._options.model_dump()
        if num_outs is not None:
            opts["num_outs"] = num_outs
        if memory_space is not None:
            opts["memory_space"] = memory_space
        if tiled is not None:
            opts["tiled"] = tiled
        if fp32_dest_acc_en is not None:
            opts["fp32_dest_acc_en"] = fp32_dest_acc_en
        if dst_full_sync_en is not None:
            opts["dst_full_sync_en"] = dst_full_sync_en
        if objective is not None:
            opts["objective"] = objective
        if placement is not None:
            opts["placement"] = placement
        self._options = ProgramOptions(**opts)
        return self

    def with_indexing(
        self,
        indexing_maps: list[Callable[..., object]],
        iterator_types: list[str],
    ) -> Self:
        """Set indexing_maps and iterator_types."""
        self._indexing_maps = indexing_maps
        self._iterator_types = iterator_types
        return self

    def build(self) -> ProgramSpec:
        """Build ProgramSpec from current builder state."""
        return ProgramSpec(
            program=self._program,
            grid=self._grid,
            options=self._options,
            indexing_maps=self._indexing_maps,
            iterator_types=self._iterator_types,
        )


__all__ = ["ProgramBuilder"]
