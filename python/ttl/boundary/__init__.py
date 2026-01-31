# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Boundary types: Protocols and opaque aliases for external libraries (ttnn, MLIR).

Any is confined to ttnn_proxy.py; other modules use these types for type safety.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from .mlir_types import MlirModuleLike
from .ttnn_types import (
    BoundingBoxLike,
    CoreRangeSetLike,
    GridSizeLike,
    TtnnDeviceLike,
    TtnnTensorLike,
)

__all__ = [
    "BoundingBoxLike",
    "CoreRangeSetLike",
    "GridSizeLike",
    "MlirModuleLike",
    "TtnnDeviceLike",
    "TtnnTensorLike",
]
