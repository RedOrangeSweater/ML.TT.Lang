#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark TTL -> TTKernel lowering time for TRID vs global barriers.

This script measures wall-clock time of invoking `ttlang-opt` on a set of MLIR
inputs using:
  - default lowering (global barriers)
  - TRID-aware lowering (`use-trid-barriers=1`)

Notes:
  - This measures end-to-end tool invocation time (process startup included).
    To reduce startup noise, use `--synthetic-copies` to generate larger inputs
    that amortize fixed overhead.
  - The simulator does not model NOC/TRID timing; this is a compiler-side
    measurement of lowering cost, not a hardware-runtime benchmark.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchResult:
    name: str
    mode: str
    iterations: int
    times_s: list[float]

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.times_s)

    @property
    def stdev_s(self) -> float:
        if len(self.times_s) < 2:
            return 0.0
        return statistics.stdev(self.times_s)

    @property
    def min_s(self) -> float:
        return min(self.times_s)

    @property
    def max_s(self) -> float:
        return max(self.times_s)


def _run_timed(cmd: list[str], env: dict[str, str]) -> float:
    start = time.perf_counter()
    subprocess.run(
        cmd,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - start


def _bench_one(
    *,
    name: str,
    mode: str,
    cmd: list[str],
    env: dict[str, str],
    warmup: int,
    iterations: int,
) -> BenchResult:
    for _ in range(warmup):
        _run_timed(cmd, env=env)

    times_s = [_run_timed(cmd, env=env) for _ in range(iterations)]
    return BenchResult(name=name, mode=mode, iterations=iterations, times_s=times_s)


def _fmt_seconds(s: float) -> str:
    # Keep stable column width without scientific notation.
    if s >= 10.0:
        return f"{s:6.2f}"
    if s >= 1.0:
        return f"{s:6.3f}"
    return f"{s:6.4f}"


def _synthetic_ttl_module(*, copies: int) -> str:
    if copies <= 0:
        raise ValueError("copies must be > 0")

    # Keep this intentionally simple: one slice + one CB reused across many copies.
    # This is a compile-time stress test for TRID allocation / barrier insertion.
    lines: list[str] = []
    lines.append("#dram = #ttnn.buffer_type<dram>")
    lines.append(
        "#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, "
        "memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>"
    )
    lines.append("module {")
    lines.append(
        "  func.func @synthetic_many_copies("
        "%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>"
        ") attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], "
        "ttl.kernel_thread = #ttkernel.thread<noc>} {"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append(
        "    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>"
    )
    lines.append(
        "    %slice = ttl.tensor_slice %arg0[%c0, %c0] : "
        "tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> "
        "tensor<1x1x!ttcore.tile<32x32, f32>, #layout>"
    )

    for i in range(copies):
        lines.append(
            f"    %xf{i} = ttl.copy %slice, %cb : "
            "(tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) "
            "-> !ttl.transfer_handle<read>"
        )

    for i in range(copies):
        lines.append(f"    ttl.wait %xf{i} : !ttl.transfer_handle<read>")

    lines.append("    func.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_inputs = [
        repo_root / "test/ttlang/Conversion/TTLToTTKernel/dma_single_core.mlir",
        repo_root / "test/ttlang/Conversion/TTLToTTKernel/loopback_dram_copy.mlir",
        repo_root / "test/ttlang/Conversion/TTLToTTKernel/trid_barriers.mlir",
    ]

    parser = argparse.ArgumentParser(
        description="Benchmark ttlang-opt lowering time (TRID vs global barriers)."
    )
    parser.add_argument(
        "--ttlang-opt",
        default=os.environ.get("TTLANG_OPT", "ttlang-opt"),
        help="Path to ttlang-opt (default: $TTLANG_OPT or ttlang-opt in PATH).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=default_inputs,
        help="MLIR input files to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations per (file, mode).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs per (file, mode).",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Disable canonicalize/cse (benchmark only convert pass).",
    )
    parser.add_argument(
        "--synthetic-copies",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Generate additional synthetic MLIR inputs with N ttl.copy/ttl.wait ops. "
            "Useful to amortize process startup and highlight per-copy overhead."
        ),
    )
    parser.add_argument(
        "--disable-mlir-threading",
        action="store_true",
        help="Pass --mlir-disable-threading for more stable timings.",
    )
    parser.add_argument(
        "--split-input-file",
        action="store_true",
        default=True,
        help="Pass --split-input-file (default: enabled).",
    )
    args = parser.parse_args()

    inputs: list[Path] = []
    for p in args.inputs:
        resolved = p if p.is_absolute() else (repo_root / p).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input not found: {resolved}")
        inputs.append(resolved)

    common_flags: list[str] = []
    if args.disable_mlir_threading:
        common_flags.append("--mlir-disable-threading")
    if args.split_input_file:
        common_flags.append("--split-input-file")
    if not args.no_canonicalize:
        common_flags += ["--canonicalize", "-cse"]

    env = dict(os.environ)
    env.setdefault("TTNN_LOG_LEVEL", "ERROR")

    temp_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.synthetic_copies:
            temp_dir_ctx = tempfile.TemporaryDirectory(prefix="ttlang_bench_trid_")
            tmpdir = Path(temp_dir_ctx.name)
            for n in args.synthetic_copies:
                out = tmpdir / f"synthetic_many_copies_{n}.mlir"
                out.write_text(_synthetic_ttl_module(copies=n), encoding="utf-8")
                inputs.append(out)

        def display_name(p: Path) -> str:
            try:
                return p.relative_to(repo_root).as_posix()
            except ValueError:
                return p.name

        rows: list[BenchResult] = []
        for input_path in inputs:
            name = display_name(input_path)

            global_cmd = [
                args.ttlang_opt,
                "--convert-ttl-to-ttkernel",
                *common_flags,
                str(input_path),
            ]
            trid_cmd = [
                args.ttlang_opt,
                "--convert-ttl-to-ttkernel=use-trid-barriers=1",
                *common_flags,
                str(input_path),
            ]

            rows.append(
                _bench_one(
                    name=name,
                    mode="global",
                    cmd=global_cmd,
                    env=env,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )
            rows.append(
                _bench_one(
                    name=name,
                    mode="trid",
                    cmd=trid_cmd,
                    env=env,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()

    by_name: dict[str, dict[str, BenchResult]] = {}
    for r in rows:
        by_name.setdefault(r.name, {})[r.mode] = r

    print("ttlang-opt lowering time (seconds)")
    print(f"iterations={args.iterations} warmup={args.warmup} canonicalize={'no' if args.no_canonicalize else 'yes'}")
    if args.disable_mlir_threading:
        print("mlir_threading=disabled")
    print()
    print(
        "file".ljust(64)
        + "  "
        + "global_med"
        + "  "
        + "trid_med"
        + "  "
        + "delta_med"
        + "  "
        + "ratio"
    )
    print("-" * (64 + 2 + 8 + 2 + 8 + 2 + 9 + 2 + 5))

    for name in sorted(by_name.keys()):
        global_r = by_name[name]["global"]
        trid_r = by_name[name]["trid"]
        delta = trid_r.median_s - global_r.median_s
        ratio = trid_r.median_s / global_r.median_s if global_r.median_s > 0 else float("inf")
        print(
            name.ljust(64)
            + "  "
            + _fmt_seconds(global_r.median_s)
            + "  "
            + _fmt_seconds(trid_r.median_s)
            + "  "
            + _fmt_seconds(delta)
            + "  "
            + f"{ratio:5.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

