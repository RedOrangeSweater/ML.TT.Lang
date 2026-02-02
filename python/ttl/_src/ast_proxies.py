# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Generic, NoReturn, TypeVar

from ttmlir.ir import Context, Location

from ..diagnostics import TTLangCompileError
from .auto_profile import LineSignpost, OpSignpost
from .context import make_file_loc

T = TypeVar("T", bound=ast.AST)

class NodeProxy(Generic[T]):
    """Base proxy for AST nodes to simplify location and error handling."""

    def __init__(self, node: T):
        self.node = node

    @property
    def lineno(self) -> int | None:
        return getattr(self.node, "lineno", None)

    @property
    def col_offset(self) -> int | None:
        return getattr(self.node, "col_offset", None)

    def location(self, ctx: Context, source_file: str | None, line_offset: int = 0) -> Location:
        """Return MLIR location for this node."""
        if source_file and self.lineno is not None:
            return make_file_loc(ctx, source_file, self.node, line_offset)
        return Location.unknown(ctx)

    def error(self, message: str, source_file: str | None, line_offset: int = 0) -> NoReturn:
        """Raise TTLangCompileError with node's location info."""
        line = (self.lineno + line_offset) if isinstance(self.lineno, int) else None
        col = (self.col_offset + 1) if isinstance(self.col_offset, int) else None
        raise TTLangCompileError(
            message,
            source_file=source_file,
            line=line,
            col=col,
        )

    def source_line(self, source_lines: list[str] | None, line_offset: int = 0) -> str:
        """Extract the text of the source line associated with this node."""
        if not isinstance(self.lineno, int):
            return "<unknown line>"

        if source_lines and 0 < self.lineno <= len(source_lines):
            return source_lines[self.lineno - 1].strip()

        file_lineno = self.lineno + line_offset
        return f"<line {file_lineno}>"

    def line_signpost(self, line_offset: int = 0) -> LineSignpost:
        """Create a LineSignpost for this node."""
        return LineSignpost(self.lineno or 0, line_offset)

    def op_signpost(self, op_name: str, line_offset: int = 0, implicit: bool = False) -> OpSignpost:
        """Create an OpSignpost for this node."""
        return OpSignpost(op_name, self.lineno or 0, line_offset, implicit)

    @property
    def is_name(self) -> bool:
        """Check if this node is an ast.Name."""
        return isinstance(self.node, ast.Name)

    @property
    def name_id(self) -> str | None:
        """Return the id if this is an ast.Name."""
        return self.node.id if isinstance(self.node, ast.Name) else None

class CallProxy(NodeProxy[ast.Call]):
    """Proxy for ast.Call nodes."""

    @property
    def func(self) -> ast.AST:
        return self.node.func

    @property
    def args(self) -> Sequence[ast.AST]:
        return self.node.args

    @property
    def keywords(self) -> Sequence[ast.keyword]:
        return self.node.keywords

    @property
    def func_name(self) -> str | None:
        """Return function name if it's a simple Name or Attribute."""
        if isinstance(self.node.func, ast.Name):
            return self.node.func.id
        if isinstance(self.node.func, ast.Attribute):
            return self.node.func.attr
        return None

class AttributeProxy(NodeProxy[ast.Attribute]):
    """Proxy for ast.Attribute nodes."""

    @property
    def value(self) -> ast.AST:
        return self.node.value

    @property
    def attr(self) -> str:
        return self.node.attr

    @property
    def is_ttl_module(self) -> bool:
        """Check if this is a ttl.XXX access."""
        return isinstance(self.node.value, ast.Name) and self.node.value.id == "ttl"

    @property
    def is_ttl_math(self) -> bool:
        """Check if this is a ttl.math.XXX access."""
        return (
            isinstance(self.node.value, ast.Attribute)
            and isinstance(self.node.value.value, ast.Name)
            and self.node.value.value.id == "ttl"
            and self.node.value.attr == "math"
        )

class AssignProxy(NodeProxy[ast.Assign]):
    """Proxy for ast.Assign nodes."""

    @property
    def targets(self) -> Sequence[ast.AST]:
        return self.node.targets

    @property
    def value(self) -> ast.AST:
        return self.node.value

    @property
    def first_target(self) -> ast.AST | None:
        return self.node.targets[0] if self.node.targets else None

class BinOpProxy(NodeProxy[ast.BinOp]):
    """Proxy for ast.BinOp nodes."""

    @property
    def left(self) -> ast.AST:
        return self.node.left

    @property
    def op(self) -> ast.operator:
        return self.node.op

    @property
    def right(self) -> ast.AST:
        return self.node.right

class SubscriptProxy(NodeProxy[ast.Subscript]):
    """Proxy for ast.Subscript nodes."""

    @property
    def value(self) -> ast.AST:
        return self.node.value

    @property
    def slice(self) -> ast.AST:
        return self.node.slice
