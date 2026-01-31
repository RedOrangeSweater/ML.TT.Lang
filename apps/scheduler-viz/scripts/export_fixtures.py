#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Export scheduler input JSON from compiled tt-lang programs (toy_program_add, etc.).

Compiles each example in compile-only mode (no device), calls kernel.get_scheduler_input()
and writes apps/scheduler-viz/fixtures/<name>.json.

Run from tt-lang repo root with venv activated:
  python apps/scheduler-viz/scripts/export_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Repo root = parent of apps/ (script is apps/scheduler-viz/scripts/export_fixtures.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Compile-only: no device required
os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch

FIXTURES_DIR = REPO_ROOT / "apps" / "scheduler-viz" / "fixtures"


def export_toy_program_add() -> None:
    from examples.toy_program_add import simple_add

    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    simple_add(lhs, rhs, out)
    kernel = getattr(simple_add, "_last_compiled_kernel", None)
    if kernel is None:
        raise RuntimeError(
            "toy_program_add: no _last_compiled_kernel (compile failed?)"
        )
    data = kernel.get_scheduler_input()
    out_path = FIXTURES_DIR / "toy_program_add.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


def export_toy_program_broadcast() -> None:
    from examples.toy_program_broadcast import fused_bcast

    a = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    b = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    c = torch.zeros((32, 32), dtype=torch.bfloat16)
    c[0, :] = 1.0
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    fused_bcast(a, b, c, out)
    kernel = getattr(fused_bcast, "_last_compiled_kernel", None)
    if kernel is None:
        raise RuntimeError(
            "toy_program_broadcast: no _last_compiled_kernel (compile failed?)"
        )
    data = kernel.get_scheduler_input()
    out_path = FIXTURES_DIR / "toy_program_broadcast.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


def export_toy_program_multicore_auto() -> None:
    # grid="auto" requires ttnn tensors/device to resolve; use stub export for fixtures
    from ttl.scheduler import export_scheduler_input

    thread_infos = [
        ("demo_compute", "compute"),
        ("demo_read", "noc"),
        ("demo_write", "noc"),
    ]
    grid = (2, 2)
    data = export_scheduler_input(thread_infos, grid, None)
    out_path = FIXTURES_DIR / "toy_program_multicore_auto.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    export_toy_program_add()
    export_toy_program_broadcast()
    export_toy_program_multicore_auto()
    print(f"Done. Fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
