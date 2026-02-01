# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compilation context: CompilerContext, TTLCompilerConfig, ThreadSourceInfo, location helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ast
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..constants import MemorySpace

if TYPE_CHECKING:
    from ..compile.source_context import CompilationSourceContext


def make_file_loc(ctx, source_file: str, node: ast.AST, line_offset: int = 0):
    """Create an MLIR file location from an AST node."""
    from ttmlir.ir import Location

    if not hasattr(node, "lineno"):
        raise ValueError(f"AST node {type(node).__name__} has no line number")
    return Location.file(
        source_file, node.lineno + line_offset, node.col_offset + 1, ctx
    )


class CompilerContext(BaseModel):
    """Immutable compilation context for TTL kernels."""

    model_config = ConfigDict(frozen=True)

    grid: list[int] = Field(..., description="Grid dimensions (cols, rows)")
    memory_space: MemorySpace = Field(..., description="L1 or DRAM")
    tiled: bool = Field(..., description="Whether to use tiled layout")


class TTLCompilerConfig(BaseModel):
    """Pydantic config for TTLGenericCompiler; built from source_context + kwargs."""

    model_config = ConfigDict(extra="ignore")

    source_context: CompilationSourceContext | None = None
    grid: list[int] = Field(default_factory=lambda: [1, 1])
    memory_space: MemorySpace = MemorySpace.L1
    tiled: bool = True
    debug_locations: bool = False
    source_file: str = Field("<unknown>", validation_alias="_source_file")
    source_lines: list[str] = Field(
        default_factory=list, validation_alias="_source_lines"
    )
    line_offset: int = Field(0, validation_alias="_line_offset")
    fn_globals: dict[str, object] = Field(
        default_factory=dict, validation_alias="_globals"
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_source_context(cls, data: object) -> object:
        """Fill source fields from source_context when provided; kwargs override."""
        if not isinstance(data, dict):
            return data
        ctx = data.get("source_context")
        if ctx is None:
            return data
        from ..compile.source_context import CompilationSourceContext

        if not isinstance(ctx, CompilationSourceContext):
            return data
        data.setdefault("_source_file", ctx.source_file)
        data.setdefault("_source_lines", ctx.source_lines)
        data.setdefault("_line_offset", ctx.line_offset)
        data.setdefault("debug_locations", ctx.debug_locations)
        data.setdefault("_globals", ctx.fn_globals)
        return data


class ThreadSourceInfo(BaseModel):
    """Structured source info for a compiled thread. Exposed via TTLGenericCompiler.source_info."""

    source_file: str = "<unknown>"
    source_lines: list[str] = Field(default_factory=list)
    line_offset: int = 0


__all__ = [
    "CompilerContext",
    "ThreadSourceInfo",
    "TTLCompilerConfig",
    "make_file_loc",
]
