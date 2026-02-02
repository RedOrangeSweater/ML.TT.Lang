# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Constants used throughout the DSL."""

from __future__ import annotations

from enum import StrEnum


class MemorySpace(StrEnum):
    """Memory space for tensors: L1, DRAM, or unknown (non-TTNN/unparseable)."""

    L1 = "L1"
    DRAM = "DRAM"
    UNKNOWN = "unknown"


class Objective(StrEnum):
    """Scheduler objective policy."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"


class Placement(StrEnum):
    """Scheduler placement policy."""

    AUTO = "auto"
    MANUAL = "manual"


# Device-supported memory spaces only (L1/DRAM).
SUPPORTED_MEMORY_SPACES: frozenset[MemorySpace] = frozenset({MemorySpace.L1, MemorySpace.DRAM})

DEFAULT_TILE_SIZE = 32
