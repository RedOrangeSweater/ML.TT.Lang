# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime layer: compilation artifacts -> descriptors -> run_kernel_on_device.

One focus: KernelSpec, CB configs, CoreRangeSet -> ttnn descriptors -> execution.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from ..kernel_runner import (
    CBDescriptorBuildRequest,
    KernelDescriptorBuildRequest,
    KernelSpec,
    RunKernelRequest,
    build_cb_descriptors,
    build_kernel_descriptors,
    build_tensor_accessor_args,
    run_kernel_on_device,
)

__all__ = [
    "CBDescriptorBuildRequest",
    "KernelDescriptorBuildRequest",
    "KernelSpec",
    "RunKernelRequest",
    "build_cb_descriptors",
    "build_kernel_descriptors",
    "build_tensor_accessor_args",
    "run_kernel_on_device",
]
