# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Measure cold vs warm "compile-to-first-run" latency for a tt-lang kernel.

This script is intended to be a practical harness for measuring how fast a new
kernel becomes runnable on device, and how that differs between cold and warm
cache states.

It relies on opt-in timing in `python/ttl/ttl_api.py`:
- TTLANG_PROFILE_COMPILE=1
- TTLANG_PROFILE_COMPILE_OUT=/path/to/out.jsonl

Usage (from repo root):

  source build/env/activate
  python tools/profile_reconfiguration_latency.py --device-id 0

Optional:
  --device-profiler      Enable TT-Metal device profiler (cycles) via env var
  --out /tmp/profile.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import ttnn  # type: ignore[import-untyped]
import ttl  # type: ignore[import-untyped]


def _to_l1(torch_tensor: torch.Tensor, device) -> ttnn.Tensor:
    """Create a TTNN tensor in L1 from a torch tensor (via DRAM first)."""
    dram_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    """Simple add kernel compiled for TTNN interop."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_out():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


def _clear_caches(device) -> None:
    """Best-effort: clear in-memory kernel cache and device program cache."""
    try:
        ttnn.device.ClearKernelCache()
    except Exception as e:
        print(f"[warn] ClearKernelCache failed: {e}")

    # MeshDevice exposes these methods via nanobind (works for single-device too).
    try:
        device.disable_and_clear_program_cache()
    except Exception:
        try:
            device.clear_program_cache()
        except Exception as e:
            print(f"[warn] clear_program_cache failed: {e}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _summarize_latest(events: list[dict[str, object]]) -> dict[str, object]:
    compile_ev = next((e for e in reversed(events) if e.get("kind") == "compile"), None)
    exec_ev = next((e for e in reversed(events) if e.get("kind") == "execute"), None)
    return {
        "compile": compile_ev,
        "execute": exec_ev,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=f"/tmp/{os.environ.get('USER', 'default')}/ttlang_reconfig_profile.jsonl",
    )
    parser.add_argument(
        "--device-profiler",
        action="store_true",
        help="Enable TT-Metal device profiler (cycles) via TT_METAL_DEVICE_PROFILER=1.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    os.environ["TTLANG_PROFILE_COMPILE"] = "1"
    os.environ["TTLANG_PROFILE_COMPILE_OUT"] = str(out_path)

    if args.device_profiler:
        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

    device = ttnn.open_device(device_id=args.device_id)
    try:
        # Allocate input/output tensors.
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out_torch = torch.full((32, 32), -999.0, dtype=torch.bfloat16)

        lhs = _to_l1(lhs_torch, device)
        rhs = _to_l1(rhs_torch, device)
        out = _to_l1(out_torch, device)

        print("=== Cold start (clear caches) ===")
        _clear_caches(device)
        t0 = time.perf_counter()
        add_kernel(lhs, rhs, out)
        ttnn.synchronize_device(device)
        cold_wall_s = time.perf_counter() - t0
        cold_events = _read_jsonl(out_path)
        cold_summary = _summarize_latest(cold_events)
        print(json.dumps({"cold_wall_s": cold_wall_s, **cold_summary}, indent=2))

        print("\n=== Warm start (same kernel, cached) ===")
        t0 = time.perf_counter()
        add_kernel(lhs, rhs, out)
        ttnn.synchronize_device(device)
        warm_wall_s = time.perf_counter() - t0
        warm_events = _read_jsonl(out_path)
        warm_summary = _summarize_latest(warm_events)
        print(json.dumps({"warm_wall_s": warm_wall_s, **warm_summary}, indent=2))

        if args.device_profiler:
            try:
                ttnn.device.ReadDeviceProfiler(device)
                print(
                    "\nDevice profiler results were collected. "
                    "See TT-Metal generated profiler logs under "
                    "${TT_METAL_HOME}/generated/profiler/.logs/."
                )
            except Exception as e:
                print(f"[warn] ReadDeviceProfiler failed: {e}")

    finally:
        ttnn.close_device(device)

    print(f"\nRaw profiling output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

