# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI-like middleware/transform framework for tt-lang Python layers.

This package provides a small core for composing transforms (middlewares) over
typed Pydantic contexts. Each layer (program/compile/runtime) can define its own
context dialect and a growing set of transforms, similar to MLIR pass pipelines.
"""

from .core import compose

__all__ = [
    "compose",
]

