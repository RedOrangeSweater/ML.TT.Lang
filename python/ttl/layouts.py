# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layout creation utilities for tensor distribution across cores."""

from ttmlir.dialects import ttcore, ttnn

from pydantic import BaseModel, ConfigDict, Field

from .constants import DEFAULT_TILE_SIZE
from .dtype_utils import TensorDtype


class TTNNLayoutConfig(BaseModel):
    """Configuration for TTNN layout creation. Supports L1/DRAM interleaved tiled layouts."""

    model_config = ConfigDict(frozen=True)

    logical_shape: list[int] = Field(
        ..., description="Logical tensor shape (rows, cols)"
    )
    grid: list[int] = Field(..., description="Grid dimensions (cols, rows)")
    dtype: str = Field(..., description="Tensor dtype (ttnn or string)")


# TTNN BufferType enum values (from TTNNOpsEnums.td)
_TTNN_BUFFER_TYPE_L1 = 1

# TTNN TensorMemoryLayout enum values (from TTNNOpsEnums.td)
_TTNN_TENSOR_MEMORY_LAYOUT_INTERLEAVED = 0


def create_ttnn_layout(ctx, config: TTNNLayoutConfig):
    """
    Create a TTNNLayoutAttr for L1 interleaved tiled tensors.

    Supports: L1/DRAM memory, Interleaved layout, tiled (32x32 tiles).

    Args:
        ctx: MLIR context
        config: Configuration with logical_shape, grid, and dtype

    Returns:
        TTNNLayoutAttr

    Raises:
        ValueError: If configuration is unsupported
    """
    if len(config.logical_shape) != 2:
        raise ValueError(f"Only 2D tensors supported, got shape {config.logical_shape}")

    if len(config.grid) != 2:
        raise ValueError(f"Only 2D grids supported, got grid {config.grid}")

    # config.grid is (cols, rows) from tt-lang API, but MLIR expects (rows, cols)
    grid_cols, grid_rows = config.grid
    mlir_grid = [grid_rows, grid_cols]

    # logical_shape is (rows, cols), mlir_grid is (rows, cols)

    ttcore_dtype = TensorDtype(dtype=config.dtype).ttcore_dtype
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    grid_attr = ttcore.ir.GridAttr.get(ctx, mlir_grid)

    return ttnn.ir.TTNNLayoutAttr.get(
        ctx,
        config.logical_shape,
        element_type,
        _TTNN_BUFFER_TYPE_L1,
        grid_attr,
        _TTNN_TENSOR_MEMORY_LAYOUT_INTERLEAVED,
    )
