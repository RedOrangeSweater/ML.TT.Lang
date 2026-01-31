# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Protocols for ttnn-facing values used in compile/runtime layers.

Minimum surface required by kernel_runner, ttl_api, program, dtype_utils.
ttnn_proxy remains the only module that returns Any (from .to_ttnn()).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GridSizeLike(Protocol):
    """Protocol for grid size (e.g. from CoreRangeSet.bounding_box().grid_size())."""

    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...


@runtime_checkable
class BoundingBoxLike(Protocol):
    """Protocol for bounding box (e.g. from CoreRangeSet.bounding_box())."""

    def grid_size(self) -> GridSizeLike: ...


@runtime_checkable
class CoreRangeSetLike(Protocol):
    """Protocol for ttnn CoreRangeSet (bounding_box, grid_size)."""

    def bounding_box(self) -> BoundingBoxLike: ...


# Alias for clarity in type hints (CoreRangeSetLike is the same).
TtnnCoreRangeSetLike = CoreRangeSetLike


@runtime_checkable
class TtnnTensorLike(Protocol):
    """Protocol for ttnn tensor: buffer_address for descriptors, device for grid."""

    def buffer_address(self) -> object: ...
    def device(self) -> TtnnDeviceLike: ...


@runtime_checkable
class TtnnDeviceLike(Protocol):
    """Protocol for ttnn device (compute_with_storage_grid_size for grid)."""

    def compute_with_storage_grid_size(self) -> GridSizeLike: ...
