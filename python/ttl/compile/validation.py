# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile-layer validation: TTNN tensor and kernel-count checks for TTNNKernelCompileRequest.

Used by TTNNKernelCompileRequest model_validator; no validate_* calls in business logic.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

from typing import Any

from ttmlir.passes import get_ttkernel_names

from ..dtype_utils import (
    detect_memory_space_from_tensor,
    is_interleaved_tensor,
    is_ttnn_tensor,
)


def validate_ttnn_tensors_for_request(args: tuple[Any, ...]) -> None:
    """Validate tensor types and TTNN tensor properties. Raises ValueError on invalid."""
    ttnn_count = sum(1 for arg in args if is_ttnn_tensor(arg))
    if ttnn_count > 0 and ttnn_count < len(args):
        raise ValueError(
            f"TTNN interop requires all tensors to be the same type. "
            f"Got {ttnn_count} TTNN tensors and {len(args) - ttnn_count} host tensors. "
            f"Mixed tensor types would generate extra bounce kernels."
        )
    for i, arg in enumerate(args):
        if not is_ttnn_tensor(arg):
            continue
        mem_space = detect_memory_space_from_tensor(arg, "unknown")
        if mem_space not in ("L1", "DRAM"):
            raise ValueError(
                f"TTNN interop requires L1 or DRAM memory space, but tensor {i} is in {mem_space}."
            )
        if not is_interleaved_tensor(arg):
            raise ValueError(
                f"TTNN interop requires interleaved tensors, but tensor {i} is not. "
                f"Use ttnn.DRAM_MEMORY_CONFIG or ttnn.L1_MEMORY_CONFIG for interleaved tensors."
            )
        if hasattr(arg, "layout") and "TILE" not in str(arg.layout):
            raise ValueError(
                f"TTNN interop requires tilized tensors, but tensor {i} has layout {arg.layout}. "
                f"Use ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) to convert."
            )


def validate_kernel_count_for_request(module: Any) -> None:
    """Validate kernel count (exactly 3: 1 compute + 2 data movement). Raises ValueError if not."""
    kernel_info = get_ttkernel_names(module)
    if len(kernel_info) != 3:
        compute_count = sum(1 for _, t in kernel_info if t == "compute")
        dm_count = sum(1 for _, t in kernel_info if t == "noc")
        raise ValueError(
            f"TTNN interop requires exactly 3 kernels (1 compute + 2 data movement), "
            f"got {len(kernel_info)} kernels ({compute_count} compute, {dm_count} data movement). "
            f"Each core has only 2 NOCs, so more than 2 DM kernels causes NOC conflicts."
        )
