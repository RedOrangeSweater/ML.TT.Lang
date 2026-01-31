# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared kernel execution logic for tt-lang.

Provides functions for building kernel descriptors, CB descriptors, and
executing kernels on device via ttnn.generic_op. Used by both the Python
DSL (CompiledTTNNKernel) and ME2E tests.

All public build/run APIs accept Pydantic request models (Runtime dialect).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import ttnn  # type: ignore[import-untyped]
except (ModuleNotFoundError, ImportError):
    ttnn = None

from pydantic import BaseModel, ConfigDict, Field

from .dtype_utils import TensorDtype


@dataclass
class KernelSpec:
    """Specification for a single kernel to execute.

    Attributes:
        path: Path to the kernel C++ source file.
        thread_type: Type of kernel ("compute", "noc", or "ethernet").
        tensor_indices: List of global tensor indices this kernel accesses.
            For DM kernels, these determine which buffer addresses go in
            common_runtime_args, in order.
        config: Kernel config descriptor (ComputeConfigDescriptor,
            ReaderConfigDescriptor, WriterConfigDescriptor, or EthernetConfigDescriptor).
    """

    path: str
    thread_type: str
    tensor_indices: list[int]
    config: Any


# -----------------------------------------------------------------------------
# Request models (Runtime dialect; single entry point per transformation)
# -----------------------------------------------------------------------------


class KernelDescriptorBuildRequest(BaseModel):
    """Request for building kernel descriptors. Replaces 7 positional parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kernel_specs: list[KernelSpec] = Field(..., description="Kernel specifications")
    tensors: list[Any] = Field(..., description="ttnn.Tensor objects; position = global index")
    tensor_accessor_args: list[int] = Field(..., description="Flattened compile-time args from all tensors")
    core_ranges: Any = Field(..., description="ttnn.CoreRangeSet for kernel execution")
    grid_cols: int = Field(..., ge=1, description="Grid columns (x dimension)")
    grid_rows: int = Field(..., ge=1, description="Grid rows (y dimension)")
    num_cbs: int = Field(..., ge=0, description="Total number of circular buffers")


class CBDescriptorBuildRequest(BaseModel):
    """Request for building circular buffer descriptors. Replaces 3 positional parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensors: list[Any] = Field(..., description="ttnn.Tensor objects; position = CB index (None for intermediate)")
    cb_configs: list[Any] = Field(..., description="CircularBuffer objects per CB index")
    core_ranges: Any = Field(..., description="ttnn.CoreRangeSet for CB allocation")


class RunKernelRequest(BaseModel):
    """Request for running kernels on device. Single entry point for run_kernel_on_device."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kernel_specs: list[KernelSpec] = Field(..., description="Kernel specifications")
    tensors: list[Any] = Field(..., description="ttnn.Tensor objects")
    cb_configs: list[Any] = Field(..., description="CircularBuffer objects per CB index")
    core_ranges: Any = Field(..., description="ttnn.CoreRangeSet for execution")
    program_hash: int | None = Field(default=None, description="Program cache hash (not yet used)")


def build_tensor_accessor_args(tensors: list[Any]) -> list[int]:
    """
    Build compile-time args for tensor accessors.

    Args:
        tensors: List of ttnn.Tensor objects on device.

    Returns:
        List of compile-time args (flattened TensorAccessorArgs for all tensors).
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    args = []
    for tensor in tensors:
        tensor_args = ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
        args.extend(tensor_args)
    return args


def build_kernel_descriptors(req: KernelDescriptorBuildRequest) -> list[Any]:
    """
    Build kernel descriptors for ttnn.generic_op.

    Args:
        req: Request with kernel_specs, tensors, tensor_accessor_args,
            core_ranges, grid_cols, grid_rows, num_cbs.

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []
    cb_indices = list(range(req.num_cbs))

    for spec in req.kernel_specs:
        runtime_args = [[[] for _ in range(req.grid_rows)] for _ in range(req.grid_cols)]
        common_runtime_args = [
            req.tensors[idx].buffer_address() for idx in spec.tensor_indices
        ]
        if spec.thread_type == "compute":
            kernel_compile_time_args = cb_indices
        else:
            kernel_compile_time_args = cb_indices + list(req.tensor_accessor_args)

        kernel_desc = ttnn.KernelDescriptor(
            kernel_source=spec.path,
            core_ranges=req.core_ranges,
            compile_time_args=kernel_compile_time_args,
            runtime_args=runtime_args,
            common_runtime_args=common_runtime_args,
            config=spec.config,
        )
        kernel_descriptors.append(kernel_desc)

    return kernel_descriptors


def build_cb_descriptors(req: CBDescriptorBuildRequest) -> list[Any]:
    """
    Build circular buffer descriptors for ttnn.generic_op.

    Args:
        req: Request with tensors, cb_configs, core_ranges.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    cb_descriptors = []
    for i, cb in enumerate(req.cb_configs):
        if cb is None:
            raise ValueError(
                f"Missing CB config for index {i}. "
                f"All CB indices must have associated CircularBuffer configurations."
            )

        # Get dtype from CB's reference tensor (ttnn boundary: resolve ttnn.DataType only here).
        ref_tensor = cb.tensor
        td = TensorDtype(dtype=ref_tensor.dtype)
        data_format = getattr(ttnn.DataType, td.dtype_name())
        page_size = td.tile_bytes()
        num_tiles = cb.shape[0] * cb.shape[1] * cb.buffer_factor
        total_size = num_tiles * page_size

        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=data_format,
            page_size=page_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=req.core_ranges,
            format_descriptors=[cb_format],
        )
        cb_descriptors.append(cb_desc)

    return cb_descriptors


def run_kernel_on_device(req: RunKernelRequest) -> Any:
    """
    Execute kernels on device using ttnn.generic_op.

    Builds all descriptors from the request and runs the program.

    Args:
        req: Request with kernel_specs, tensors, cb_configs, core_ranges, program_hash.

    Returns:
        Result from ttnn.generic_op (typically None or output tensor).
    """
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    tensor_accessor_args = build_tensor_accessor_args(req.tensors)
    grid_size = req.core_ranges.bounding_box().grid_size()
    grid_cols = grid_size.x
    grid_rows = grid_size.y

    kernel_req = KernelDescriptorBuildRequest(
        kernel_specs=req.kernel_specs,
        tensors=req.tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=req.core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_cbs=len(req.cb_configs),
    )
    kernel_descriptors = build_kernel_descriptors(kernel_req)

    cb_req = CBDescriptorBuildRequest(
        tensors=req.tensors,
        cb_configs=req.cb_configs,
        core_ranges=req.core_ranges,
    )
    cb_descriptors = build_cb_descriptors(cb_req)

    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
    )
    return ttnn.generic_op(list(req.tensors), program)


__all__ = [
    "KernelSpec",
    "KernelDescriptorBuildRequest",
    "CBDescriptorBuildRequest",
    "RunKernelRequest",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "run_kernel_on_device",
]
