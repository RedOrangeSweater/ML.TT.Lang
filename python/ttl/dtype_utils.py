# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch, TT core, and dtype names.

Core has no ttnn dependency. Conversions use ttcore.DataType and canonical
dtype names (DTypeName), aligned with tt-metal DataType enum. All coercion
is done via Pydantic validators (TTCoreDataTypeLike, DTypeNameLike, TileBytesLike).
The ttnn boundary (e.g. kernel_runner) resolves ttnn.DataType via
getattr(ttnn.DataType, dtype_name) when needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Union

import torch
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, computed_field

from ttmlir.dialects import ttcore

from .constants import MemorySpace
from .runtime_tensor import is_ttnn_tensor

_INTERLEAVED = "INTERLEAVED"


# -----------------------------------------------------------------------------
# Canonical dtype names (mirror tt-metal DataType enum; no ttnn import).
# Optional FLOAT16 for ttcore/MLIR; Bool -> UINT8.
# -----------------------------------------------------------------------------
class DTypeName(str, Enum):
    """Canonical dtype names. Matches tt-metal DataType enum for boundary getattr(ttnn.DataType, name)."""

    BFLOAT16 = "BFLOAT16"
    FLOAT32 = "FLOAT32"
    UINT32 = "UINT32"
    BFLOAT8_B = "BFLOAT8_B"
    BFLOAT4_B = "BFLOAT4_B"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    INT32 = "INT32"
    INVALID = "INVALID"
    # Optional for ttcore/MLIR (not in tt-metal DataType):
    FLOAT16 = "FLOAT16"


# Single source of truth: (ttcore, name, tile_bytes). Build _TTCORE_TO_NAME, _NAME_TO_TTCORE, _TTCORE_TO_TILE_BYTES.
_DTYPE_TABLE: list[tuple[ttcore.DataType, str, int]] = [
    (ttcore.DataType.Float32, DTypeName.FLOAT32.value, 32 * 32 * 4),
    (ttcore.DataType.Float16, DTypeName.FLOAT16.value, 32 * 32 * 2),
    (ttcore.DataType.BFloat16, DTypeName.BFLOAT16.value, 32 * 32 * 2),
    (ttcore.DataType.Int32, DTypeName.INT32.value, 32 * 32 * 4),
    (ttcore.DataType.UInt32, DTypeName.UINT32.value, 32 * 32 * 4),
    (ttcore.DataType.UInt16, DTypeName.UINT16.value, 32 * 32 * 2),
    (ttcore.DataType.UInt8, DTypeName.UINT8.value, 32 * 32),
    (ttcore.DataType.Bool, DTypeName.UINT8.value, 32 * 32),  # bool -> UINT8 tile size
    (
        ttcore.DataType.BFloat16,
        DTypeName.BFLOAT8_B.value,
        32 * 32 * 2,
    ),  # approximate for MLIR
    (
        ttcore.DataType.BFloat16,
        DTypeName.BFLOAT4_B.value,
        32 * 32 * 2,
    ),  # approximate for MLIR
]

_NAME_TO_TTCORE: dict[str, ttcore.DataType] = {}
_TTCORE_TO_NAME: dict[ttcore.DataType, str] = {}
_TTCORE_TO_TILE_BYTES: dict[ttcore.DataType, int] = {}
for tc, name, tb in _DTYPE_TABLE:
    if name not in _NAME_TO_TTCORE:
        _NAME_TO_TTCORE[name] = tc
    if tc not in _TTCORE_TO_NAME:
        _TTCORE_TO_NAME[tc] = name  # first occurrence per ttcore (canonical name)
    _TTCORE_TO_TILE_BYTES[tc] = tb


# -----------------------------------------------------------------------------
# TTNN memory config proxy: Pydantic model deriving memory_space and is_interleaved
# -----------------------------------------------------------------------------
class TTNNMemoryConfigProxy(BaseModel):
    """Pydantic proxy for TTNN tensor memory config.

    Holds tensor and default; derives memory_space from tensor.memory_config().buffer_type
    and is_interleaved from tensor.memory_config().memory_layout. All string parsing
    (L1/DRAM/INTERLEAVED) lives inside this model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensor: object = Field(..., description="TTNN tensor or any object (non-TTNN uses default).")
    default: MemorySpace = Field(default="unknown", description="MemorySpace when not TTNN or unparseable.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def memory_space(self) -> MemorySpace:
        if not is_ttnn_tensor(self.tensor):
            return self.default
        mem_config = self.tensor.memory_config()
        if hasattr(mem_config, "buffer_type"):
            s = str(mem_config.buffer_type)
            if "L1" in s:
                return "L1"
            if "DRAM" in s:
                return "DRAM"
        return self.default

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_interleaved(self) -> bool:
        if not is_ttnn_tensor(self.tensor):
            return False
        mem_config = self.tensor.memory_config()
        if hasattr(mem_config, "memory_layout"):
            return _INTERLEAVED in str(mem_config.memory_layout)
        return False


def torch_dtype_to_ttcore_datatype(torch_dtype: torch.dtype) -> ttcore.DataType:
    """
    Convert PyTorch dtype to ttcore.DataType enum.

    Args:
        torch_dtype: PyTorch dtype (torch.float32, torch.int32, etc.)

    Returns:
        ttcore.DataType enum value

    Raises:
        ValueError: If dtype is not supported
    """
    if torch_dtype == torch.float32:
        return ttcore.DataType.Float32
    if torch_dtype == torch.float16:
        return ttcore.DataType.Float16
    if torch_dtype == torch.bfloat16:
        return ttcore.DataType.BFloat16
    if torch_dtype == torch.int32:
        return ttcore.DataType.Int32
    if torch_dtype == torch.uint32:
        return ttcore.DataType.UInt32
    if torch_dtype == torch.uint16:
        return ttcore.DataType.UInt16
    if torch_dtype == torch.uint8:
        return ttcore.DataType.UInt8
    if torch_dtype == torch.bool:
        return ttcore.DataType.Bool

    raise ValueError(f"Unsupported torch dtype for ttcore.DataType: {torch_dtype}")


def _name_to_ttcore(name: str) -> ttcore.DataType:
    """Resolve dtype name (e.g. 'FLOAT32') to ttcore.DataType."""
    key = name.upper()
    if key not in _NAME_TO_TTCORE:
        raise ValueError(f"Unsupported dtype name for ttcore.DataType: {name}")
    return _NAME_TO_TTCORE[key]


# -----------------------------------------------------------------------------
# Pydantic coercion: one entry point per target type
# -----------------------------------------------------------------------------


def _coerce_to_ttcore(
    v: Union[torch.dtype, ttcore.DataType, str, object],
) -> ttcore.DataType:
    """Coerce torch/ttcore/name/object-with-.name to ttcore.DataType. No ttnn."""
    if isinstance(v, ttcore.DataType):
        return v
    if isinstance(v, torch.dtype):
        return torch_dtype_to_ttcore_datatype(v)
    if isinstance(v, str):
        return _name_to_ttcore(v)
    # Accept any object with .name that matches a canonical dtype (e.g. ttnn.DataType at boundary).
    if hasattr(v, "name"):
        name = getattr(v, "name")
        if isinstance(name, str) and name.upper() in _NAME_TO_TTCORE:
            return _NAME_TO_TTCORE[name.upper()]
    raise ValueError(f"Cannot coerce to ttcore.DataType: {v}")


def _coerce_to_dtype_name(
    v: Union[torch.dtype, ttcore.DataType, str, DTypeName, object],
) -> DTypeName:
    """Coerce to DTypeName. No ttnn."""
    if isinstance(v, DTypeName):
        return v
    if isinstance(v, str):
        key = v.upper()
        try:
            return DTypeName(key)
        except ValueError:
            raise ValueError(f"Unsupported dtype name for DTypeName: {v}") from None
    if isinstance(v, ttcore.DataType):
        if v in _TTCORE_TO_NAME:
            return DTypeName(_TTCORE_TO_NAME[v])
        raise ValueError(f"No DTypeName for ttcore.DataType: {v}")
    if isinstance(v, torch.dtype):
        tc = _coerce_to_ttcore(v)
        return DTypeName(_TTCORE_TO_NAME[tc])
    if hasattr(v, "name"):
        name = getattr(v, "name")
        if isinstance(name, str):
            return DTypeName(name.upper())
    raise ValueError(f"Cannot coerce to DTypeName: {v}")


def _coerce_to_tile_bytes(v: Union[torch.dtype, ttcore.DataType, str, object]) -> int:
    """Coerce to tile size in bytes. No ttnn."""
    tc = _coerce_to_ttcore(v)
    if tc in _TTCORE_TO_TILE_BYTES:
        return _TTCORE_TO_TILE_BYTES[tc]
    raise ValueError(f"Unsupported dtype for tile size: {tc}")


TTCoreDataTypeLike = Annotated[ttcore.DataType, BeforeValidator(_coerce_to_ttcore)]
DTypeNameLike = Annotated[DTypeName, BeforeValidator(_coerce_to_dtype_name)]
TileBytesLike = Annotated[int, BeforeValidator(_coerce_to_tile_bytes)]


# -----------------------------------------------------------------------------
# TensorDtype: Pydantic model; field uses TTCoreDataTypeLike so coercion is automatic
# -----------------------------------------------------------------------------


class TensorDtype(BaseModel):
    """
    Pydantic wrapper for tensor dtype (torch, ttcore, canonical name, or object with .name).

    Field dtype is TTCoreDataTypeLike: Pydantic coerces on assignment.
    Exposes .ttcore_dtype, .dtype_name(), .tile_bytes(). No ttnn dependency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype: TTCoreDataTypeLike = Field(
        ...,
        description="ttcore.DataType (coerced from torch/name/object with .name via TTCoreDataTypeLike)",
    )

    @property
    def ttcore_dtype(self) -> ttcore.DataType:
        """Return ttcore.DataType."""
        return self.dtype

    def dtype_name(self) -> str:
        """Return canonical dtype name (e.g. 'FLOAT32') for use at ttnn boundary."""
        return _TTCORE_TO_NAME[self.dtype]

    def tile_bytes(self) -> int:
        """Return tile size in bytes from table."""
        return _TTCORE_TO_TILE_BYTES[self.dtype]
