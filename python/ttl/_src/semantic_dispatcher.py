# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .ttl_ast import TTLGenericCompiler

_T = TypeVar("_T")

class SemanticDispatcher:
    """
    Dispatches AST nodes to handlers based on semantic rules.
    Allows decoupling complex logic from the main visitor.
    """

    def __init__(self, compiler: TTLGenericCompiler):
        self.compiler = compiler
        self.handlers: list[tuple[Callable[[ast.AST], bool], Callable[[ast.AST], Any]]] = []

    def register(self, predicate: Callable[[ast.AST], bool], handler: Callable[[ast.AST], Any]):
        """Register a handler for nodes that match the predicate."""
        self.handlers.append((predicate, handler))

    def dispatch(self, node: ast.AST) -> Any:
        """Find and call the first matching handler for the node."""
        for predicate, handler in self.handlers:
            if predicate(node):
                return handler(node)
        return None
