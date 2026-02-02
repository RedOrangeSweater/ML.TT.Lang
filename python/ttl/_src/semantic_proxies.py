# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, cast

from .ast_proxies import CallProxy, NodeProxy

if TYPE_CHECKING:
    from .ttl_ast import TTLGenericCompiler

class SemanticProxy(NodeProxy):
    """Base class for semantic proxies that understand the 'meaning' of code."""
    pass

class CircularBufferOpProxy(SemanticProxy):
    """Proxy for CircularBuffer operations (wait, reserve, push, pop)."""
    
    @classmethod
    def try_wrap(cls, node: ast.AST) -> CircularBufferOpProxy | None:
        if not isinstance(node, ast.Call):
            return None
        
        call = CallProxy(node)
        if isinstance(call.func, ast.Attribute) and call.func.attr in ("wait", "reserve", "push", "pop"):
            # Check if the base object is a CB (this might require compiler context)
            return cls(node)
        return None

    @property
    def op_name(self) -> str:
        return cast(ast.Attribute, cast(ast.Call, self.node).func).attr

    @property
    def cb_var_name(self) -> str | None:
        func = cast(ast.Call, self.node).func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id
        return None

class WithCBProxy(SemanticProxy):
    """Proxy for 'with cb.wait() as data' pattern."""
    
    def __init__(self, item: ast.withitem):
        super().__init__(item.context_expr)
        self.item = item

    @property
    def call(self) -> CallProxy:
        return CallProxy(cast(ast.Call, self.item.context_expr))

    @property
    def cb_var_name(self) -> str:
        return cast(ast.Name, cast(ast.Attribute, self.call.func).value).id

    @property
    def method_name(self) -> str:
        return cast(ast.Attribute, self.call.func).attr

    @property
    def optional_name(self) -> str | None:
        if isinstance(self.item.optional_vars, ast.Name):
            return self.item.optional_vars.id
        return None
