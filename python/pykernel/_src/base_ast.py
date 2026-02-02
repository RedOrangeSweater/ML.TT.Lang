# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file was copied from tt-mlir/tools/pykernel/_src/base_ast.py

from __future__ import annotations

import ast
import contextlib
import inspect
from collections.abc import Generator
from typing import Any

from ttmlir.dialects import emitc, func
from ttmlir.ir import *


class Scope:
    """Manages symbol table scoping."""

    def __init__(self):
        self.symbol_tables: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def new_scope(self) -> Generator[None]:
        """Context manager to manage symbol table scoping."""
        self.symbol_tables.append({})
        try:
            yield
        finally:
            self.symbol_tables.pop()

    @property
    def current_scope(self) -> dict[str, Any]:
        """Return the current (innermost) symbol table."""
        if not self.symbol_tables:
            raise RuntimeError("No active scope in symbol table")
        return self.symbol_tables[-1]

    def define(self, name: str, value: Any) -> None:
        """Define a variable in the current scope."""
        self.current_scope[name] = value

    def lookup(self, name: str) -> Any | None:
        """Look up a variable in all active scopes, starting from the innermost."""
        for sym_table in reversed(self.symbol_tables):
            if name in sym_table:
                return sym_table[name]
        return None

    def get_table_with_var(self, var_name: str) -> dict[str, Any] | None:
        """Return the symbol table containing the variable, or None if not found."""
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return None


class PyKernelAstBase(ast.NodeVisitor):
    _fn_map = {}

    def __init__(self, *args, **kwargs):
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.scope = Scope()
        self.supported_nodes = [ast.Module, ast.Return, ast.Expr]

    def _get_source_comment(self, node):
        """
        Retrieve the source snippet corresponding to the given node and format it as comments.

        This function extracts the relevant lines of source code using the node's location
        attributes (lineno, end_lineno, col_offset, end_col_offset), prefixes each line with
        '//', and returns the concatenated snippet as a single string.

        Args:
            node: An AST node that contains information about the source code segment location.

        Returns:
            str: The snippet of source code formatted with '//' at the beginning of each line.
        """
        result = ""
        if self.verbose and self.source_code:
            for i in range(node.lineno - 1, node.end_lineno):
                result += (
                    "// "
                    + self.source_code[i][node.col_offset : node.end_col_offset]
                    + "\n"
                )
        return result.strip()

    def _get_source_comment_block(self, node, delim: str = "):"):
        """
        Generates a comment block extracted from the source code related to the given AST node.

        This function examines lines of source code starting at node.lineno and continuing up to
        node.end_lineno, looking for the specified delimiter. Each line is prefixed with "// " to form
        a comment block. If the delimiter is found, it stops appending further lines.

        Args:
            node: An AST node that provides line number boundaries (lineno, end_lineno) for source extraction.
            delim (str): The string delimiter to indicate where to stop collecting lines. Defaults to "):".

        Returns:
            str: A multi-line comment string containing the relevant source code lines, each prefixed with "// ".
        """
        result = ""
        if self.verbose and self.source_code:
            idx = node.lineno - 1
            result = "// "
            while idx < node.end_lineno:
                line = self.source_code[idx]
                end_pattern = line.find(delim)
                if end_pattern != -1:
                    # First occurence of end_pattern detected, save the current splice of the string + exist
                    result += line[: end_pattern + 2].lstrip()
                    break
                idx += 1
                result += f"{line}\n// "
        return result

    def visit_Module(self, node):
        # Set default basic block
        with InsertionPoint(self.insert_point), Location.unknown():
            with self.scope.new_scope():
                for stmt in node.body:
                    self.visit(stmt)

    def visit_Return(self, node):
        # TODO: handle more than one return, i.e. tuples, expressions etc.
        if node.value:
            # Visit the return value and return it
            return_value = self.visit(node.value)
            func.ReturnOp([return_value])
        else:
            # Empty return
            func.ReturnOp([])

    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit(self, node: ast.AST, **kwargs):
        if not any(isinstance(node, sn) for sn in self.supported_nodes):
            raise NotImplementedError(f"visit {type(node).__name__} not supported")

        if self.verbose and isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            # Create a verbatim Op here to store the comment
            source_code = self._get_source_comment(node)
            emitc.verbatim(source_code, [])

        # Figure out which node to visit. Not using super().visit() in order to pass kwargs.
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)

        params = inspect.signature(visitor).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        return visitor(node, **filtered_kwargs) if filtered_kwargs else visitor(node)
