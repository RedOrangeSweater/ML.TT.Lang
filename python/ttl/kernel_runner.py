# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Thin re-export of Runtime layer for backward compatibility.

Implementation lives in ttl.runtime.runner. Use ttl.runtime or ttl.kernel_runner.
"""

from .runtime import (
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
