# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Сценарии симуляции:
- baseline vs ping-pong (2 слота)
- baseline vs reader/compute/writer pipeline (CB pages)
"""

from __future__ import annotations

# NOTE: salabim does not ship type stubs; keep this file out of strict typing.
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUntypedBaseClass=false

from dataclasses import dataclass

import salabim as sim  # pyright: ignore[reportMissingImports]

from tools.scheduling_sim.models import (
    BusyResource,
    LatencyConfig,
    LevelBuffer,
    SimResult,
)


@dataclass(frozen=True)
class SimConfig:
    tiles: int = 32
    seed: int = 1
    noc_read_capacity: int = 1
    noc_write_capacity: int = 1
    compute_capacity: int = 1
    cb_in_pages: int = 2
    cb_out_pages: int = 2
    lat: LatencyConfig = LatencyConfig()


class _Metrics:
    def __init__(self) -> None:
        self.compute_busy = 0.0
        self.compute_idle = 0.0


def _make_env(seed: int) -> sim.Environment:
    # We use generator-based process() with `yield`, so disable yieldless mode.
    sim.yieldless(False)
    env = sim.Environment(trace=False)
    env.random_seed(seed)
    return env


def run_pingpong_baseline(cfg: SimConfig) -> SimResult:
    """Baseline: каждый тайл: DMA(read) -> wait -> compute (один слот)."""

    env = _make_env(cfg.seed)
    lat = cfg.lat

    noc_read = BusyResource("noc_read", env=env, capacity=cfg.noc_read_capacity)
    compute = BusyResource("compute", env=env, capacity=cfg.compute_capacity)
    m = _Metrics()

    class Driver(sim.Component):
        def process(self):
            for _ in range(cfg.tiles):
                # DMA read
                yield self.request(noc_read)
                noc_read.note_acquire()
                yield self.hold(lat.dma_read_lat)
                noc_read.note_release()
                self.release(noc_read)

                # Compute consumes data
                yield self.request(compute)
                compute.note_acquire()
                yield self.hold(lat.compute_lat)
                compute.note_release()
                self.release(compute)

    Driver()
    env.run()

    # BusyResource уже накапливает busy_time; idle оценим как makespan - busy (capacity=1)
    t_end = float(env.now())
    m.compute_busy = compute.busy_time
    m.compute_idle = max(0.0, t_end - m.compute_busy)

    return SimResult(
        scenario="pingpong_baseline",
        tiles=cfg.tiles,
        t_end=t_end,
        compute_busy=m.compute_busy,
        compute_idle=m.compute_idle,
        noc_read_busy=noc_read.busy_time,
        noc_write_busy=0.0,
        cb_in_max_level=0,
        cb_out_max_level=0,
    )


def run_pingpong_two_slots(cfg: SimConfig) -> SimResult:
    """Ping-pong: 2 слота. Пока compute ест slot[k], prefetchит slot[k^1]."""

    env = _make_env(cfg.seed)
    lat = cfg.lat

    noc_read = BusyResource("noc_read", env=env, capacity=cfg.noc_read_capacity)
    compute = BusyResource("compute", env=env, capacity=cfg.compute_capacity)

    class Slot:
        def __init__(self, name: str):
            self.name = name
            self.ready = sim.State(name=f"{name}.ready")
            self.ready.set(False)
            self.in_use = sim.State(name=f"{name}.in_use")
            self.in_use.set(False)

    slots = [Slot("slot0"), Slot("slot1")]

    def prefetch(slot: Slot) -> sim.Component:
        class Prefetch(sim.Component):
            def process(self):
                yield self.request(noc_read)
                noc_read.note_acquire()
                yield self.hold(lat.dma_read_lat)
                noc_read.note_release()
                self.release(noc_read)
                slot.ready.set(True)

        return Prefetch()

    class Driver(sim.Component):
        def process(self):
            # Prefetch first two tiles (if exist).
            tiles_total = cfg.tiles
            next_tile = 0
            for s in slots:
                if next_tile >= tiles_total:
                    break
                prefetch(s)
                next_tile += 1

            # Consume tiles in order, ping-pong slots.
            for i in range(tiles_total):
                s = slots[i % 2]
                # Wait until this slot ready
                yield self.wait(s.ready)
                s.in_use.set(True)
                s.ready.set(False)

                # Start prefetch for the next tile into the freed slot (asap)
                if next_tile < tiles_total:
                    prefetch(
                        s
                    )  # reuse same slot after compute; models "fill in parallel"
                    next_tile += 1

                # Compute
                yield self.request(compute)
                compute.note_acquire()
                yield self.hold(lat.compute_lat)
                compute.note_release()
                self.release(compute)
                s.in_use.set(False)

    Driver()
    env.run()

    t_end = float(env.now())
    compute_busy = compute.busy_time
    compute_idle = max(0.0, t_end - compute_busy)

    return SimResult(
        scenario="pingpong_two_slots",
        tiles=cfg.tiles,
        t_end=t_end,
        compute_busy=compute_busy,
        compute_idle=compute_idle,
        noc_read_busy=noc_read.busy_time,
        noc_write_busy=0.0,
        cb_in_max_level=0,
        cb_out_max_level=0,
    )


def run_rcw_baseline(cfg: SimConfig) -> SimResult:
    """
    Baseline RCW без перекрытия (по тайлу):
    Для каждого тайла последовательно выполняем:
      reader(DRAM->CB_in) -> compute(CB_in->CB_out) -> writer(CB_out->DRAM)

    Это baseline "без pipeline" при ограниченной емкости CB.
    """

    env = _make_env(cfg.seed)
    lat = cfg.lat

    noc_read = BusyResource("noc_read", env=env, capacity=cfg.noc_read_capacity)
    noc_write = BusyResource("noc_write", env=env, capacity=cfg.noc_write_capacity)
    compute = BusyResource("compute", env=env, capacity=cfg.compute_capacity)

    cb_in = LevelBuffer(env, capacity_pages=cfg.cb_in_pages, name="cb_in")
    cb_out = LevelBuffer(env, capacity_pages=cfg.cb_out_pages, name="cb_out")

    class Driver(sim.Component):
        def process(self):
            for _ in range(cfg.tiles):
                # Reader: DRAM -> CB_in
                yield self.wait(cb_in.wait_can_push())
                yield self.request(noc_read)
                noc_read.note_acquire()
                yield self.hold(lat.dma_read_lat)
                noc_read.note_release()
                self.release(noc_read)
                cb_in.push()
                yield self.hold(lat.cb_push_lat)

                # Compute: CB_in -> CB_out
                yield self.wait(cb_in.wait_can_pop())
                cb_in.pop()
                yield self.hold(lat.cb_pop_lat)

                yield self.wait(cb_out.wait_can_push())
                yield self.request(compute)
                compute.note_acquire()
                yield self.hold(lat.compute_lat)
                compute.note_release()
                self.release(compute)
                cb_out.push()
                yield self.hold(lat.cb_push_lat)

                # Writer: CB_out -> DRAM
                yield self.wait(cb_out.wait_can_pop())
                cb_out.pop()
                yield self.hold(lat.cb_pop_lat)

                yield self.request(noc_write)
                noc_write.note_acquire()
                yield self.hold(lat.dma_write_lat)
                noc_write.note_release()
                self.release(noc_write)

    Driver()
    env.run()

    t_end = float(env.now())
    compute_busy = compute.busy_time
    compute_idle = max(0.0, t_end - compute_busy)

    return SimResult(
        scenario="rcw_baseline",
        tiles=cfg.tiles,
        t_end=t_end,
        compute_busy=compute_busy,
        compute_idle=compute_idle,
        noc_read_busy=noc_read.busy_time,
        noc_write_busy=noc_write.busy_time,
        cb_in_max_level=cb_in.max_level,
        cb_out_max_level=cb_out.max_level,
    )


def run_rcw_pipeline(cfg: SimConfig) -> SimResult:
    """Pipeline RCW: reader/compute/writer как 3 компонента, параллельно через CB."""

    env = _make_env(cfg.seed)
    lat = cfg.lat

    noc_read = BusyResource("noc_read", env=env, capacity=cfg.noc_read_capacity)
    noc_write = BusyResource("noc_write", env=env, capacity=cfg.noc_write_capacity)
    compute = BusyResource("compute", env=env, capacity=cfg.compute_capacity)

    cb_in = LevelBuffer(env, capacity_pages=cfg.cb_in_pages, name="cb_in")
    cb_out = LevelBuffer(env, capacity_pages=cfg.cb_out_pages, name="cb_out")

    class Reader(sim.Component):
        def process(self):
            for _ in range(cfg.tiles):
                yield self.wait(cb_in.wait_can_push())
                yield self.request(noc_read)
                noc_read.note_acquire()
                yield self.hold(lat.dma_read_lat)
                noc_read.note_release()
                self.release(noc_read)
                cb_in.push()
                yield self.hold(lat.cb_push_lat)

    class Compute(sim.Component):
        def process(self):
            for _ in range(cfg.tiles):
                yield self.wait(cb_in.wait_can_pop())
                cb_in.pop()
                yield self.hold(lat.cb_pop_lat)

                yield self.wait(cb_out.wait_can_push())
                yield self.request(compute)
                compute.note_acquire()
                yield self.hold(lat.compute_lat)
                compute.note_release()
                self.release(compute)
                cb_out.push()
                yield self.hold(lat.cb_push_lat)

    class Writer(sim.Component):
        def process(self):
            for _ in range(cfg.tiles):
                yield self.wait(cb_out.wait_can_pop())
                cb_out.pop()
                yield self.hold(lat.cb_pop_lat)

                yield self.request(noc_write)
                noc_write.note_acquire()
                yield self.hold(lat.dma_write_lat)
                noc_write.note_release()
                self.release(noc_write)

    Reader()
    Compute()
    Writer()
    env.run()

    t_end = float(env.now())
    compute_busy = compute.busy_time
    compute_idle = max(0.0, t_end - compute_busy)

    return SimResult(
        scenario="rcw_pipeline",
        tiles=cfg.tiles,
        t_end=t_end,
        compute_busy=compute_busy,
        compute_idle=compute_idle,
        noc_read_busy=noc_read.busy_time,
        noc_write_busy=noc_write.busy_time,
        cb_in_max_level=cb_in.max_level,
        cb_out_max_level=cb_out.max_level,
    )
