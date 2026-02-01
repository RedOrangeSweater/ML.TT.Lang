# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL DSL module providing the unified ttl.* API namespace.

Decorators:
    @ttl.program() - Define a program (may compile to one or more device kernels)
    @ttl.kernel()  - Alias for program; backward compatibility
    @ttl.compute() - Define a compute thread (auto-collected)
    @ttl.datamovement() - Define a data movement thread (auto-collected)

Functions:
    ttl.make_circular_buffer_like() - Create a circular buffer
    ttl.copy() - Asynchronous data transfer
    ttl.core(dims=2) - Get current core's coordinates as (x, y) tuple
    ttl.grid_size(dims=2) - Get grid size as (x_size, y_size) tuple

Math operations:
    ttl.math.sqrt(), ttl.math.exp(), etc.
"""

# Math operations namespace
from . import ttl_math as math
from .operators import copy, core, grid_size
from .ttl_api import (
    Program,
    ProgramSpec,
    compute,
    datamovement,
    program,
    run,
)
from .ttl_api import (
    pykernel_gen as kernel,
)

__all__ = [
    "Program",
    "ProgramSpec",
    "compute",
    "copy",
    "core",
    "datamovement",
    "grid_size",
    "kernel",
    "math",
    "program",
    "run",
]
