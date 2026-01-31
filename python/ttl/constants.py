# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Constants used throughout the DSL."""

from __future__ import annotations

from typing import Literal

MemorySpace = Literal["L1", "DRAM"]
SUPPORTED_MEMORY_SPACES: frozenset[MemorySpace] = frozenset({"L1", "DRAM"})

DEFAULT_TILE_SIZE = 32
