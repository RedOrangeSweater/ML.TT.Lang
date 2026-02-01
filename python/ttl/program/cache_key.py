# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Cache key for compile cache: tensor properties and runtime compute options."""

from __future__ import annotations

from ..constants import MemorySpace
from ..dtype_utils import TTNNMemoryConfigProxy, is_ttnn_tensor


def get_tensor_cache_info(tensor: object) -> tuple[tuple[int, ...], str, MemorySpace, str]:
    """Extract cache-relevant info from a tensor: (shape, dtype, memory_space, layout)."""
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    proxy = TTNNMemoryConfigProxy(tensor=tensor, default=MemorySpace.UNKNOWN)
    memory_space = proxy.memory_space
    layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
    return (shape, dtype, memory_space, layout)


def make_cache_key(
    args: tuple[object, ...],
    fp32_dest_acc_en: bool | None,
    dst_full_sync_en: bool | None,
) -> tuple[object, ...]:
    """Create cache key from tensor properties and runtime compute config parameters."""
    tensor_key = tuple(
        get_tensor_cache_info(arg) for arg in args if is_ttnn_tensor(arg)
    )
    return (tensor_key, fp32_dest_acc_en, dst_full_sync_en)
