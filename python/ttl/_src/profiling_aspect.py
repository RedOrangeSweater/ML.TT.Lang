# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import contextlib
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, TypeVar

from .ast_proxies import NodeProxy
from .auto_profile import Signpost, SignpostBoundary

if TYPE_CHECKING:
    from .ttl_ast import TTLGenericCompiler

_T = TypeVar("_T")

class ProfilingAspect:
    """
    Aspect layer for auto-profiling.
    Separates profiling instrumentation from core MLIR generation.
    """

    def __init__(self, compiler: TTLGenericCompiler):
        self.compiler = compiler
        self._current_line_signpost: Signpost | None = None

    @property
    def enabled(self) -> bool:
        return self.compiler.auto_profile_enabled

    def emit_line_signpost_if_needed(self, node: ast.AST) -> None:
        """Emit signposts at line boundaries for auto-profiling."""
        if not self.enabled:
            return

        proxy = NodeProxy(node)
        if not isinstance(proxy.lineno, int):
            return

        signpost = proxy.line_signpost(self.compiler.config.line_offset)
        if self._current_line_signpost and self._current_line_signpost.file_lineno == signpost.file_lineno:
            return

        if self._current_line_signpost:
            self.compiler._emit_marker(self._current_line_signpost, SignpostBoundary.AFTER)

        self.compiler._register_pair(signpost, proxy.source_line(self.compiler.config.source_lines, self.compiler.config.line_offset))
        self.compiler._emit_marker(signpost, SignpostBoundary.BEFORE)
        self._current_line_signpost = signpost

    def close_final_signpost(self) -> None:
        """Close the final signpost at the end of function body."""
        if self.enabled and self._current_line_signpost:
            self.compiler._emit_marker(self._current_line_signpost, SignpostBoundary.AFTER)
            self._current_line_signpost = None

    @contextlib.contextmanager
    def instrument_op(
        self,
        node: ast.AST,
        op_name: str,
        implicit: bool = False,
    ) -> Generator[None]:
        """Context manager to instrument an operation with signposts."""
        if not self.enabled:
            with self.compiler._loc_for_node(node):
                yield
            return

        proxy = NodeProxy(node)
        signpost = proxy.op_signpost(op_name, self.compiler.config.line_offset, implicit)

        if not signpost.is_valid:
            with self.compiler._loc_for_node(node):
                yield
            return

        self.compiler._register_pair(signpost, proxy.source_line(self.compiler.config.source_lines, self.compiler.config.line_offset))

        with self.compiler._loc_for_node(node):
            self.compiler._emit_marker(signpost, SignpostBoundary.BEFORE)
            yield
            self.compiler._emit_marker(signpost, SignpostBoundary.AFTER)

    def wrap_visit(self, node: ast.AST, visit_fn: Callable[[], _T]) -> _T:
        """Wrap a visitor call with line signposting."""
        self.emit_line_signpost_if_needed(node)
        return visit_fn()
