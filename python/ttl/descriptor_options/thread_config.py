# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Thread config: ComputeConfigOptions, NocConfigOptions, ReaderConfigOptions, ThreadConfigBuildRequest."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..ttnn_proxy import (
    ComputeConfigProxy,
    ComputeConfigResolved,
    ComputeDescriptorBuildContext,
    ReaderConfigProxy,
    WriterConfigProxy,
)


class ComputeConfigOptions(BaseModel):
    """Options for building ttnn.ComputeConfigDescriptor. Proxy for fp32_dest_acc_en / dst_full_sync_en."""

    fp32_dest_acc_en: bool | None = Field(
        default=None, description="Enable fp32 destination accumulator"
    )
    dst_full_sync_en: bool | None = Field(
        default=None, description="Enable destination full sync"
    )

    def build_ttnn_descriptor(
        self, context: ComputeDescriptorBuildContext
    ) -> tuple[object, dict[str, str]]:
        """
        Build ttnn.ComputeConfigDescriptor and thread_to_kernel entries.
        When fp32_dest_acc_en is None and context.has_f32 is True, enables fp32_dest_acc_en automatically.
        """
        proxy = ComputeConfigProxy(
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
        )
        resolved = ComputeConfigResolved.from_proxy_and_context(proxy, context)
        config = resolved.build_ttnn()
        entries = {
            "TRISC_0": context.kernel_name,
            "TRISC_1": context.kernel_name,
            "TRISC_2": context.kernel_name,
        }
        return config, entries


class NocConfigOptions(BaseModel):
    """Options for building ttnn Reader/Writer config. noc_kernel_idx 0 -> Reader, 1 -> Writer."""

    noc_kernel_idx: int = Field(
        ..., ge=0, le=1, description="0=Reader/NCRISC, 1=Writer/BRISC"
    )
    kernel_name: str = Field(
        ..., description="Kernel name for thread_to_kernel entries"
    )

    def build_ttnn_descriptor(self) -> tuple[object, dict[str, str]]:
        """Build ttnn.ReaderConfigDescriptor or WriterConfigDescriptor and thread_to_kernel entries."""
        if self.noc_kernel_idx == 0:
            config = ReaderConfigProxy().build_ttnn()
            entries = {"NCRISC": self.kernel_name}
        else:
            config = WriterConfigProxy().build_ttnn()
            entries = {"BRISC": self.kernel_name}
        return config, entries


class ReaderConfigOptions(BaseModel):
    """Options for building ttnn.ReaderConfigDescriptor (fallback for unknown thread_type)."""

    kernel_name: str = Field(
        default="", description="Kernel name for thread_to_kernel entries"
    )

    def build_ttnn_descriptor(self) -> tuple[object, dict[str, str]]:
        """Build ttnn.ReaderConfigDescriptor and empty entries (default reader fallback)."""
        config = ReaderConfigProxy().build_ttnn()
        return config, {}


class ThreadConfigBuildRequest(BaseModel):
    """Request for building config for one thread. Replaces 6 parameters in _build_config_for_thread."""

    thread_type: str = Field(..., description="compute | noc | other")
    name: str = Field(..., description="Kernel name")
    noc_kernel_idx: int = Field(
        default=0, ge=0, description="NOC index (0=Reader, 1=Writer)"
    )
    compute_opts: ComputeConfigOptions = Field(
        default_factory=ComputeConfigOptions,
        description="Compute options when thread_type is compute",
    )
    has_f32: bool = Field(default=False, description="Whether args have float32")
    verbose: bool = Field(default=False, description="Print messages")

    def build_config_and_entries(self) -> tuple[object, dict[str, str]]:
        """Build (ttnn config descriptor, thread_to_kernel entries) for this thread. Single entry point."""
        if self.thread_type == "compute":
            context = ComputeDescriptorBuildContext(
                kernel_name=self.name,
                has_f32=self.has_f32,
                verbose=self.verbose,
            )
            return self.compute_opts.build_ttnn_descriptor(context)
        if self.thread_type == "noc":
            return NocConfigOptions(
                noc_kernel_idx=self.noc_kernel_idx,
                kernel_name=self.name,
            ).build_ttnn_descriptor()
        return ReaderConfigOptions(kernel_name=self.name).build_ttnn_descriptor()


__all__ = [
    "ComputeConfigOptions",
    "NocConfigOptions",
    "ReaderConfigOptions",
    "ThreadConfigBuildRequest",
]
