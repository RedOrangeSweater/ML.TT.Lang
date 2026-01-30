# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Thin profiling wrappers for tt-lang (host-side).

Design goals:
- Avoid ad-hoc perf_counter blocks sprinkled across the codebase.
- Keep overhead near-zero when disabled.
- Prefer Tracy/TT-Metal profiling as the primary source of truth, by emitting
  host-side Tracy zones via `ttnn.profiler` bindings when available.

Enablement:
- TTLANG_TRACY=1: emit Tracy zones (host-side) via `ttnn.profiler`.

Notes:
- This module is intentionally small. It is not a general telemetry framework.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator

try:
    import ttnn  # type: ignore[import-untyped]
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    ttnn = None


def tracy_enabled() -> bool:
    return os.environ.get("TTLANG_TRACY", "0") == "1" and ttnn is not None


@dataclass
class SpanResult:
    name: str
    elapsed_s: float = 0.0


@contextlib.contextmanager
def span(name: str, *, color: int = 0) -> Iterator[SpanResult]:
    """
    Create a profiling span.

    When `TTLANG_TRACY=1` and `ttnn.profiler` is available, emits a Tracy zone.

    Args:
        name: Stable span name (prefer dot-separated).
        color: Optional Tracy color (int).
    """
    res = SpanResult(name=name)
    t0 = time.perf_counter()

    started_tracy = False
    if tracy_enabled():
        try:
            # Best-effort: get source/line/function for Tracy zone metadata.
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            source = caller.f_code.co_filename if caller is not None else "<unknown>"
            funct = name
            line = caller.f_lineno if caller is not None else 0
            ttnn.profiler.start_tracy_zone(source, funct, line, color)  # type: ignore[attr-defined]
            started_tracy = True
        except Exception:
            started_tracy = False

    try:
        yield res
    finally:
        res.elapsed_s = time.perf_counter() - t0
        if started_tracy:
            try:
                ttnn.profiler.stop_tracy_zone(name, color)  # type: ignore[attr-defined]
            except Exception:
                pass


def profiled(name: str, *, color: int = 0) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator to wrap a function in a profiling span."""

    def _decorator(fn: Callable[..., object]) -> Callable[..., object]:
        def _wrapped(*args: object, **kwargs: object) -> object:
            with span(name, color=color):
                return fn(*args, **kwargs)

        return _wrapped

    return _decorator

