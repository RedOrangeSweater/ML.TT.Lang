# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Type builder: MLIR tensor type construction for ttnn tensors and layout."""

from __future__ import annotations

import ast

from ttmlir.dialects import ttcore
from ttmlir.ir import RankedTensorType

from ..constants import DEFAULT_TILE_SIZE, SUPPORTED_MEMORY_SPACES, MemorySpace
from ..diagnostics import TTLangCompileError
from ..dtype_utils import TensorDtype
from ..layouts import TTNNLayoutConfig, create_ttnn_layout
from .tensor_registry import get_tensor_source


def get_annotation_name(annotation: ast.expr) -> str:
    """Extract the type name from an annotation node.

    Handles both simple names (CircularBuffer) and qualified names (ttl.CircularBuffer).
    Returns the simple type name (e.g., 'CircularBuffer') in both cases.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    raise TypeError(f"Unsupported annotation type: {type(annotation)}")


def raise_tensor_error(tensor: object, message: str) -> None:
    """Raise TTLangCompileError with tensor source location if available."""
    source_info = get_tensor_source(tensor)
    if source_info:
        source_file, line = source_info
        raise TTLangCompileError(message, source_file=source_file, line=line)
    raise ValueError(message)


def build_tensor_type(ctx, tensor, grid, tiled: bool, memory_space: MemorySpace):
    """Build MLIR tensor type for a ttnn tensor with TTNNLayoutAttr."""
    if not tiled:
        raise ValueError("Only tiled tensors supported for TTNN interop")
    if memory_space not in SUPPORTED_MEMORY_SPACES:
        raise ValueError(
            f"Only L1 or DRAM memory space supported, got {memory_space}"
        )
    if len(grid) != 2:
        raise ValueError(f"Only 2D grids supported, got grid {tuple(grid)}")
    if len(tensor.shape) != 2:
        raise_tensor_error(
            tensor, f"Only 2D tensors supported, got shape {tensor.shape}"
        )

    tensor_rows, tensor_cols = tensor.shape

    layout = create_ttnn_layout(
        ctx,
        TTNNLayoutConfig(
            logical_shape=tensor.shape,
            grid=grid,
            dtype=tensor.dtype,
        ),
    )

    ttcore_dtype = TensorDtype(dtype=tensor.dtype).ttcore_dtype
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    total_row_tiles = (tensor_rows + DEFAULT_TILE_SIZE - 1) // DEFAULT_TILE_SIZE
    total_col_tiles = (tensor_cols + DEFAULT_TILE_SIZE - 1) // DEFAULT_TILE_SIZE
    device_shape = [total_row_tiles, total_col_tiles]

    return RankedTensorType.get(device_shape, element_type, layout)


__all__ = [
    "build_tensor_type",
    "get_annotation_name",
    "raise_tensor_error",
]
