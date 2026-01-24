# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLI для запуска сравнения:
- baseline vs ping-pong
- baseline vs reader/compute/writer pipeline

Запуск:
  source build/env/activate
  python tools/scheduling_sim/run.py --all
"""

from __future__ import annotations

import argparse

from tools.scheduling_sim.models import LatencyConfig, SimResult
from tools.scheduling_sim.scenarios import (
    SimConfig,
    run_pingpong_baseline,
    run_pingpong_two_slots,
    run_rcw_baseline,
    run_rcw_pipeline,
)


def _fmt_row(r: SimResult) -> str:
    return (
        f"{r.scenario:18} tiles={r.tiles:4d}  "
        f"t_end={r.t_end:8.2f}  "
        f"compute_idle={100.0 * r.compute_idle_frac:6.2f}%  "
        f"nocR_busy={r.noc_read_busy:8.2f}  "
        f"nocW_busy={r.noc_write_busy:8.2f}  "
        f"cb_in_max={r.cb_in_max_level:3d}  cb_out_max={r.cb_out_max_level:3d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="salabim scheduling micro-simulator")
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenario", choices=["pingpong", "rcw", "all"], default="all")

    ap.add_argument("--dma-read-lat", type=float, default=20.0)
    ap.add_argument("--dma-write-lat", type=float, default=20.0)
    ap.add_argument("--compute-lat", type=float, default=5.0)
    ap.add_argument("--cb-pages-in", type=int, default=2)
    ap.add_argument("--cb-pages-out", type=int, default=2)

    args = ap.parse_args()

    lat = LatencyConfig(
        dma_read_lat=args.dma_read_lat,
        dma_write_lat=args.dma_write_lat,
        compute_lat=args.compute_lat,
    )
    cfg = SimConfig(
        tiles=args.tiles,
        seed=args.seed,
        cb_in_pages=args.cb_pages_in,
        cb_out_pages=args.cb_pages_out,
        lat=lat,
    )

    results: list[SimResult] = []

    if args.scenario in ("pingpong", "all"):
        results.append(run_pingpong_baseline(cfg))
        results.append(run_pingpong_two_slots(cfg))

    if args.scenario in ("rcw", "all"):
        results.append(run_rcw_baseline(cfg))
        results.append(run_rcw_pipeline(cfg))

    print()
    print("=== Results ===")
    for r in results:
        print(_fmt_row(r))

    # Simple pairwise deltas (where applicable)
    def by_name(name: str) -> SimResult | None:
        for r in results:
            if r.scenario == name:
                return r
        return None

    print()
    print("=== Deltas ===")
    a = by_name("pingpong_baseline")
    b = by_name("pingpong_two_slots")
    if a and b:
        print(
            f"pingpong: t_end {a.t_end:.2f} -> {b.t_end:.2f} "
            f"({(a.t_end - b.t_end):.2f} better)"
        )

    a = by_name("rcw_baseline")
    b = by_name("rcw_pipeline")
    if a and b:
        print(
            f"rcw:     t_end {a.t_end:.2f} -> {b.t_end:.2f} "
            f"({(a.t_end - b.t_end):.2f} better)"
        )

    print()
    print("Tip: vary --dma-read-lat/--compute-lat to see when pipeline helps most.")


if __name__ == "__main__":
    main()

