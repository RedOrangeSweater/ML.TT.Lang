# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real implementation of bar: loaded on first use. Can call other lazy or existing functions."""


def bar(msg: str) -> str:
    """Return greeting. Loaded when boot code first calls bar."""
    return f"Hello, {msg}!"
