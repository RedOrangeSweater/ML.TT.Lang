# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sync primitives: SyncPrimitive, Semaphore.

Synchronization units (semaphores, barriers) composable into programs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import BuildContext, Primitive


class SyncPrimitive(BaseModel):
    """Base Pydantic model for synchronization primitives."""

    name: str = Field(default="", description="Sync point name for diagnostics")

    def build(self, ctx: BuildContext) -> None:
        """Emit sync ops into context. Override in subclasses."""
        pass

    def validate(self) -> None:
        """Validate sync config. Override in subclasses."""
        pass


class Semaphore(SyncPrimitive):
    """Semaphore primitive for producer-consumer sync."""

    sem_id: int = Field(default=0, ge=0, description="Semaphore index")


__all__ = ["Semaphore", "SyncPrimitive"]
