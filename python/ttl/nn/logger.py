# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Logger protocol and minimal implementations (jsonl + stdout) for ttl.nn.Trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Protocol for Trainer loggers. log_metrics writes step metrics."""

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """Log a dict of metrics (e.g. latency_ms, throughput). step and prefix optional."""
        ...

    def flush(self) -> None:
        """Flush any buffered output."""
        ...


class StdoutLogger:
    """Minimal logger: print metrics to stdout (one line per log_metrics call)."""

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        parts = [f"step={step}"] if step is not None else []
        if prefix:
            parts.append(f"prefix={prefix}")
        parts.append(json.dumps(metrics))
        print(" ".join(parts))

    def flush(self) -> None:
        pass


class JsonlLogger:
    """Logger that appends one JSON object per log_metrics call to a file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        record: dict[str, Any] = dict(metrics)
        if step is not None:
            record["step"] = step
        if prefix:
            record["prefix"] = prefix
        self._file.write(json.dumps(record, default=str) + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
