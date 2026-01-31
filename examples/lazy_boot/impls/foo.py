# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real implementation of foo: loaded on first use."""


def foo(x: int, y: int) -> int:
    """Return x + y. Loaded when boot code first calls foo."""
    return x + y
