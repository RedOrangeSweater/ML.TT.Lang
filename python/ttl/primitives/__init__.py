# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Composable primitives: DM, compute, sync units for PyTorch-like modular programs.

Re-exports base, dm, compute, sync, registry. See docs/sdlc/00_Main/02_Architecture/
20_IdealDataFlowAndModuleStructure.md (§7).
"""

from __future__ import annotations

from .base import BuildContext, ComposablePrimitive, Primitive
from .compute import ComputePrimitive, ElementwiseOp, MatmulOp
from .dm import DMPrimitive, Reader, Writer
from .registry import PrimitiveRegistry
from .sync import Semaphore, SyncPrimitive

__all__ = [
    "BuildContext",
    "ComposablePrimitive",
    "ComputePrimitive",
    "DMPrimitive",
    "ElementwiseOp",
    "MatmulOp",
    "Primitive",
    "PrimitiveRegistry",
    "Reader",
    "Semaphore",
    "SyncPrimitive",
    "Writer",
]
