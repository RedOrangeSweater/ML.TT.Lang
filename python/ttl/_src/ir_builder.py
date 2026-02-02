# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ttmlir.dialects import arith, func, ttcore, ttkernel
from ttmlir.ir import (
    Context,
    F32Type,
    IndexType,
    IntegerType,
    Location,
)

if TYPE_CHECKING:
    from .ttl_ast import TTLGenericCompiler

class IRBuilder:
    """
    Fluent API for building MLIR operations.
    Makes MLIR emission look like Python code.
    """

    def __init__(self, compiler: TTLGenericCompiler):
        self.compiler = compiler
        self.ctx = compiler.ctx

    @property
    def loc(self) -> Location:
        return self.compiler.loc

    def const_i64(self, value: int) -> Any:
        """Emit an i64 constant."""
        return arith.ConstantOp(IntegerType.get_signless(64, self.ctx), value).result

    def const_f32(self, value: float) -> Any:
        """Emit an f32 constant."""
        return arith.ConstantOp(F32Type.get(self.ctx), value).result

    def const_index(self, value: int) -> Any:
        """Emit an index constant."""
        return arith.ConstantOp(IndexType.get(self.ctx), value).result

    def index_cast(self, value: Any) -> Any:
        """Cast a value to index type."""
        return arith.IndexCastOp(IndexType.get(self.ctx), value).result

    def add_i64(self, lhs: Any, rhs: Any) -> Any:
        """Emit an i64 addition."""
        return arith.AddIOp(lhs, rhs).result

    def sub_i64(self, lhs: Any, rhs: Any) -> Any:
        """Emit an i64 subtraction."""
        return arith.SubIOp(lhs, rhs).result

    def mul_i64(self, lhs: Any, rhs: Any) -> Any:
        """Emit an i64 multiplication."""
        return arith.MulIOp(lhs, rhs).result
