# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic proxies for ttnn primitives. All construction of ttnn config/Core* goes through these proxies via .to_ttnn()."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _import_ttnn():  # noqa: ANN202
    try:
        import ttnn  # type: ignore[import-untyped]
        return ttnn
    except (ModuleNotFoundError, ImportError):
        raise RuntimeError("ttnn is required to build config descriptors") from None


# -----------------------------------------------------------------------------
# ComputeConfigDescriptor proxy (optional flags -> resolved -> to_ttnn)
# -----------------------------------------------------------------------------


class ComputeDescriptorBuildContext(BaseModel):
    """Context for building compute config descriptor. Supplies has_f32 and verbose for resolution."""

    kernel_name: str = Field(..., description="Kernel name for thread_to_kernel entries")
    has_f32: bool = Field(default=False, description="Whether args contain float32 (auto-enable fp32_dest_acc_en)")
    verbose: bool = Field(default=False, description="Print auto-enable messages")


class ComputeConfigProxy(BaseModel):
    """Pydantic proxy for ttnn.ComputeConfigDescriptor. Optional flags only."""

    fp32_dest_acc_en: bool | None = Field(default=None, description="Enable fp32 destination accumulator")
    dst_full_sync_en: bool | None = Field(default=None, description="Enable destination full sync")


class ComputeConfigResolved(BaseModel):
    """Resolved compute config (all bools set). Single place for .to_ttnn() mapping."""

    fp32_dest_acc_en: bool = Field(default=False)
    dst_full_sync_en: bool = Field(default=False)

    @classmethod
    def from_proxy_and_context(
        cls,
        proxy: ComputeConfigProxy,
        context: ComputeDescriptorBuildContext,
    ) -> ComputeConfigResolved:
        """Resolve optional flags from proxy using context (e.g. has_f32 -> fp32_dest_acc_en when None)."""
        fp32 = proxy.fp32_dest_acc_en
        if fp32 is None and context.has_f32:
            fp32 = True
            if context.verbose:
                print("  [fp32 detected] Enabling fp32_dest_acc_en for compute kernel")
        dst_sync = proxy.dst_full_sync_en
        return cls(
            fp32_dest_acc_en=fp32 if fp32 is not None else False,
            dst_full_sync_en=dst_sync if dst_sync is not None else False,
        )

    def to_ttnn(self) -> Any:
        """Build ttnn.ComputeConfigDescriptor from resolved fields. No branching."""
        ttnn = _import_ttnn()
        config = ttnn.ComputeConfigDescriptor()
        config.fp32_dest_acc_en = self.fp32_dest_acc_en
        config.dst_full_sync_en = self.dst_full_sync_en
        return config


# -----------------------------------------------------------------------------
# ReaderConfigDescriptor / WriterConfigDescriptor proxies
# -----------------------------------------------------------------------------


class ReaderConfigProxy(BaseModel):
    """Pydantic proxy for ttnn.ReaderConfigDescriptor (empty struct)."""

    def to_ttnn(self) -> Any:
        """Build ttnn.ReaderConfigDescriptor."""
        ttnn = _import_ttnn()
        return ttnn.ReaderConfigDescriptor()


class WriterConfigProxy(BaseModel):
    """Pydantic proxy for ttnn.WriterConfigDescriptor (empty struct)."""

    def to_ttnn(self) -> Any:
        """Build ttnn.WriterConfigDescriptor."""
        ttnn = _import_ttnn()
        return ttnn.WriterConfigDescriptor()


# -----------------------------------------------------------------------------
# CoreCoord, CoreRange, CoreRangeSet proxies
# -----------------------------------------------------------------------------


class CoreCoordProxy(BaseModel):
    """Pydantic proxy for ttnn.CoreCoord."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)

    def to_ttnn(self) -> Any:
        """Build ttnn.CoreCoord."""
        ttnn = _import_ttnn()
        return ttnn.CoreCoord(self.x, self.y)


class CoreRangeProxy(BaseModel):
    """Pydantic proxy for ttnn.CoreRange."""

    start: CoreCoordProxy = Field(...)
    end: CoreCoordProxy = Field(...)

    def to_ttnn(self) -> Any:
        """Build ttnn.CoreRange."""
        ttnn = _import_ttnn()
        return ttnn.CoreRange(self.start.to_ttnn(), self.end.to_ttnn())


class CoreRangeSetProxy(BaseModel):
    """Pydantic proxy for ttnn.CoreRangeSet. Build from grid (cols, rows)."""

    grid: tuple[int, int] = Field(
        ...,
        description="(cols, rows) matching tt-metal CoreCoord convention",
    )

    @field_validator("grid")
    @classmethod
    def grid_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        cols, rows = v
        if cols < 1 or rows < 1:
            raise ValueError("grid (cols, rows) must have both >= 1")
        return v

    def to_ttnn(self) -> Any:
        """Build ttnn.CoreRangeSet covering [0,0] to (grid[0]-1, grid[1]-1)."""
        cols, rows = self.grid
        start = CoreCoordProxy(x=0, y=0)
        end = CoreCoordProxy(x=cols - 1, y=rows - 1)
        core_range = CoreRangeProxy(start=start, end=end)
        ttnn = _import_ttnn()
        return ttnn.CoreRangeSet([core_range.to_ttnn()])
