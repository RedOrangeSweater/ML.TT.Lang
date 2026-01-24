# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Минимальные строительные блоки для дискретно-событийной симуляции на salabim.

Цель: получить понятный симулятор "DMA <-> compute" и "reader/compute/writer",
который можно расширять (в т.ч. под RL) без привязки к ttsim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import salabim as sim  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class LatencyConfig:
    """Параметры задержек (в условных 'циклах')."""

    dma_read_lat: float = 20.0
    dma_write_lat: float = 20.0
    compute_lat: float = 5.0
    cb_push_lat: float = 0.1
    cb_pop_lat: float = 0.1


@dataclass
class SimResult:
    """Сводка результатов одного прогона."""

    scenario: str
    tiles: int
    t_end: float
    compute_busy: float
    compute_idle: float
    noc_read_busy: float
    noc_write_busy: float
    cb_in_max_level: int
    cb_out_max_level: int

    @property
    def compute_idle_frac(self) -> float:
        denom = self.compute_busy + self.compute_idle
        return (self.compute_idle / denom) if denom else 0.0


class BusyResource(sim.Resource):
    """Resource с учётом 'busy time' как суммы времени владения."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.busy_time: float = 0.0
        self._last_acquire_t: float | None = None

    def note_acquire(self) -> None:
        self._last_acquire_t = float(self.env.now())

    def note_release(self) -> None:
        if self._last_acquire_t is None:
            return
        self.busy_time += float(self.env.now()) - self._last_acquire_t
        self._last_acquire_t = None


class LevelBuffer:
    """Простая модель CB как буфера с уровнем заполнения и емкостью."""

    def __init__(self, env: sim.Environment, capacity_pages: int, name: str):
        self.env = env
        self.capacity_pages = capacity_pages
        self.name = name
        self.level: int = 0
        self.max_level: int = 0

        # Условия ожидания (простая блокировка по уровню).
        self._can_push = sim.State(name=f"{name}.can_push")
        self._can_pop = sim.State(name=f"{name}.can_pop")
        self._update_states()

    def _update_states(self) -> None:
        self.max_level = max(self.max_level, self.level)
        self._can_push.set(self.level < self.capacity_pages)
        self._can_pop.set(self.level > 0)

    def push(self) -> None:
        if self.level >= self.capacity_pages:
            raise RuntimeError(f"{self.name}: overflow")
        self.level += 1
        self._update_states()

    def pop(self) -> None:
        if self.level <= 0:
            raise RuntimeError(f"{self.name}: underflow")
        self.level -= 1
        self._update_states()

    def wait_can_push(self) -> sim.State:
        return self._can_push

    def wait_can_pop(self) -> sim.State:
        return self._can_pop

