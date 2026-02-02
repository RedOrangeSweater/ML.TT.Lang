# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import NoReturn, TypeVar

from ttmlir.dialects import arith, func, ttcore, ttkernel
from ttmlir.ir import (
    Block,
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    RankedTensorType,
    SymbolTable,
)

from pykernel._src.kernel_ast import TTCompilerBase

from ..constants import DEFAULT_TILE_SIZE
from ..diagnostics import TTLangCompileError
from ..dialects import ttl
from ..dtype_utils import TensorDtype, is_ttnn_tensor
from ..ttl_utils import get_thread_type_string
from .auto_profile import (
    get_line_mapper,
    is_auto_profile_enabled,
)
from .ast_proxies import (
    NodeProxy,
    CallProxy,
    AttributeProxy,
    AssignProxy,
    BinOpProxy,
    SubscriptProxy,
)
from .context import (
    CompilerContext,
    ThreadSourceInfo,
    TTLCompilerConfig,
)
from .context import (
    make_file_loc as _make_file_loc,
)
from .tensor_registry import get_tensor_global_index
from .type_builder import build_tensor_type as _build_tensor_type

_T = TypeVar("_T")


class TTLGenericCompiler(TTCompilerBase):
    """Compiler that generates TTL dialect ops from Python AST."""

    _syntax: dict[str, Callable] = {}

    def __init__(
        self,
        name,
        kernel_type=None,
        captures=None,
        *args,
        config: TTLCompilerConfig | None = None,
        **kwargs,
    ):
        super().__init__(name, kernel_type, *args, **kwargs)
        self.loc = Location.name(self.name)
        self.captures = captures if captures is not None else {}
        self.streams: set[str] = set()
        self.supported_nodes.append(ast.AsyncFunctionDef)
        self.supported_nodes.append(ast.With)

        if config is None:
            config = TTLCompilerConfig.model_validate(kwargs)
        self.context = CompilerContext(
            grid=config.grid,
            memory_space=config.memory_space,
            tiled=config.tiled,
        )
        self.debug_locations = config.debug_locations
        self.source_file = config.source_file
        self.source_lines = config.source_lines
        self.line_offset = config.line_offset
        self.fn_globals = config.fn_globals

        self._cb_info: list[dict[str, object]] = []
        self.auto_profile_enabled = is_auto_profile_enabled()
        self.line_mapper = get_line_mapper() if self.auto_profile_enabled else None
        if self.line_mapper:
            self.line_mapper.line_offset = self.line_offset
        self._current_signpost_line: int | None = None
        self._fn_map = dict(TTLGenericCompiler._syntax)

    @property
    def source_info(self) -> ThreadSourceInfo:
        """Structured source info for error reporting and profiling."""
        return ThreadSourceInfo(
            source_file=self.source_file,
            source_lines=self.source_lines,
            line_offset=self.line_offset,
        )

    def visit_Assign(self, node):
        """Handle tuple unpacking for TTL functions like core(dims=2)."""
        proxy = AssignProxy(node)
        target = proxy.first_target
        if not isinstance(target, ast.Tuple):
            return super().visit_Assign(node)

        value = self.visit(proxy.value)
        if not isinstance(value, tuple):
            return super().visit_Assign(node)

        targets = target.elts
        if len(value) != len(targets):
            raise ValueError(
                f"Cannot unpack {len(value)} values into {len(targets)} variables"
            )

        sym_table = self.symbol_tables[-1]
        for elt, val in zip(targets, value, strict=False):
            if not isinstance(elt, ast.Name):
                raise ValueError("Tuple unpacking requires simple variable names")
            sym_table[elt.id] = val

    def _loc_for_node(self, node: ast.AST) -> Location:
        """Return file location for node if debug_locations enabled, else name location."""
        if self.debug_locations:
            proxy = NodeProxy(node)
            if proxy.lineno is not None:
                return proxy.location(self.ctx, self.source_file, self.line_offset)
        return self.loc

    def _raise_error(self, node: ast.AST, message: str) -> NoReturn:
        """Raise a TTLangCompileError with source location from AST node."""
        NodeProxy(node).error(message, self.source_file, self.line_offset)

    # Auto-profiling helpers for line-based signposting

    def _emit_signpost(self, name: str):
        """Emit a signpost operation into the MLIR."""
        ttl.signpost(name)

    def _register_signpost_pair(
        self,
        before_name: str,
        after_name: str,
        file_lineno: int,
        source_line: str,
    ) -> None:
        """Register before/after signpost pair if line_mapper is active."""
        if self.line_mapper:
            self.line_mapper.register_signpost(before_name, file_lineno, source_line)
            self.line_mapper.register_signpost(after_name, file_lineno, source_line)

    def _emit_line_signpost_if_needed(self, node: ast.AST) -> None:
        """Emit signposts at line boundaries for auto-profiling."""
        proxy = NodeProxy(node)
        lineno = proxy.lineno
        if not self.auto_profile_enabled or not isinstance(lineno, int):
            return

        file_lineno = lineno + self.line_offset
        if self._current_signpost_line == file_lineno:
            return

        if self._current_signpost_line is not None:
            self._emit_signpost(f"line_{self._current_signpost_line}_after")

        if self.source_lines and 0 < lineno <= len(self.source_lines):
            source_line = self.source_lines[lineno - 1].strip()
        else:
            source_line = f"<line {file_lineno}>"

        before_name = f"line_{file_lineno}_before"
        after_name = f"line_{file_lineno}_after"

        self._register_signpost_pair(before_name, after_name, file_lineno, source_line)

        self._emit_signpost(before_name)
        self._current_signpost_line = int(file_lineno)

    def _close_final_signpost(self):
        """Close the final signpost at the end of function body."""
        if self.auto_profile_enabled and self._current_signpost_line is not None:
            self._emit_signpost(f"line_{self._current_signpost_line}_after")
            self._current_signpost_line = None

    def _try_emit_auto_signposts(self, node: ast.AST, visit_fn: Callable[[], _T]) -> _T:
        """Emit line-based signposts if auto-profiling is enabled."""
        self._emit_line_signpost_if_needed(node)
        return visit_fn()

    def _emit_op_signposts(
        self,
        op_name: str,
        node: ast.AST,
        op_fn: Callable[[], _T],
        implicit: bool = False,
    ) -> _T:
        """Emit signposts for CB operations with op name included."""
        proxy = NodeProxy(node)
        if not self.auto_profile_enabled:
            with self._loc_for_node(node):
                return op_fn()

        lineno = proxy.lineno
        if not isinstance(lineno, int):
            with self._loc_for_node(node):
                return op_fn()

        file_lineno = lineno + self.line_offset
        prefix = "implicit_" if implicit else ""
        before_name = f"line_{file_lineno}_{prefix}{op_name}_before"
        after_name = f"line_{file_lineno}_{prefix}{op_name}_after"

        if self.source_lines and 0 < lineno <= len(self.source_lines):
            source_line = self.source_lines[lineno - 1].strip()
        else:
            source_line = f"<line {file_lineno}>"

        self._register_signpost_pair(before_name, after_name, file_lineno, source_line)

        with self._loc_for_node(node):
            self._emit_signpost(before_name)
            result = op_fn()
            self._emit_signpost(after_name)
        return result

    def visit_Call(self, node):
        """Override to set location context, catch errors, and inject auto-profiling."""
        proxy = CallProxy(node)
        with self._loc_for_node(node):
            try:
                return self._try_emit_auto_signposts(
                    node, lambda: super(TTLGenericCompiler, self).visit_Call(node)
                )
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                proxy.error(str(e), self.source_file, self.line_offset)

    def visit_BinOp(self, node):
        """Override to inject auto-profiling and provide better error messages."""
        proxy = BinOpProxy(node)
        with self._loc_for_node(node):
            try:
                return self._try_emit_auto_signposts(
                    node, lambda: super(TTLGenericCompiler, self).visit_BinOp(node)
                )
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                proxy.error(str(e), self.source_file, self.line_offset)

    def visit_Name(self, node):
        """Override to check function globals for simple constants."""
        result = super().visit_Name(node)
        if result is not None:
            return result

        # Check if it's a module-level constant
        var_name = node.id
        if var_name in self.fn_globals:
            val = self.fn_globals[var_name]
            if isinstance(val, int):
                return arith.ConstantOp(
                    IntegerType.get_signless(64, self.ctx), val
                ).result
            if isinstance(val, float):
                return arith.ConstantOp(F32Type.get(self.ctx), val).result

        return None

    def _is_ttl_module_access(self, node: ast.Attribute) -> bool:
        """Check if node is ttl.XXX access pattern."""
        return isinstance(node.value, ast.Name) and node.value.id == "ttl"

    def _is_ttl_math_access(self, node):
        """Check if node is ttl.math.XXX access pattern."""
        return (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ttl"
            and node.value.attr == "math"
        )

    def _resolve_ttl_function(
        self,
        node: ast.Attribute,
        func_args: list[object],
        kwargs: dict[str, object],
    ) -> object | None:
        """Resolve and call a ttl.XXX or ttl.math.XXX function."""
        if self._is_ttl_module_access(node):
            namespace = "ttl"
        elif self._is_ttl_math_access(node):
            namespace = "ttl.math"
        else:
            return None

        fn = self._fn_map.get(node.attr)
        if fn is None:
            self._raise_error(node, f"Unknown function: {namespace}.{node.attr}")
        return fn(*func_args, **kwargs)

    def visit_Attribute(
        self,
        node: ast.Attribute,
        func_args: list[object] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> object | None:
        """Override to set location context and catch errors for method calls."""
        proxy = AttributeProxy(node)
        func_args = [] if func_args is None else func_args
        kwargs = {} if kwargs is None else kwargs
        with self._loc_for_node(node):
            try:
                # Handle ttl.XXX and ttl.math.XXX attribute access
                if self._is_ttl_module_access(node) or self._is_ttl_math_access(node):
                    return self._resolve_ttl_function(node, func_args, kwargs)
                return super().visit_Attribute(node, func_args, kwargs)
            except (ValueError, TypeError, NotImplementedError) as e:
                if isinstance(e, TTLangCompileError):
                    raise
                proxy.error(str(e), self.source_file, self.line_offset)

    def visit_Subscript(self, node):
        """Handle tensor[row, col] or tensor[r0:r1, c0:c1] indexing."""
        proxy = SubscriptProxy(node)
        if not isinstance(proxy.value, ast.Name):
            proxy.error("TTL only supports subscripting simple variables", self.source_file, self.line_offset)

        var_name = proxy.value.id
        tbl = self._var_exists(var_name)
        if not tbl:
            proxy.error(f"Unknown variable: {var_name}", self.source_file, self.line_offset)

        tensor = tbl[var_name]
        if not isinstance(getattr(tensor, "type", None), RankedTensorType):
            proxy.error("TTL only supports subscripting tensors", self.source_file, self.line_offset)

        if isinstance(proxy.slice, ast.Tuple):
            indices = [self._build_index_or_range(elt) for elt in proxy.slice.elts]
        else:
            indices = [self._build_index_or_range(proxy.slice)]

        return (tensor, indices)

    def _to_index_value(self, node):
        """Convert AST node to MLIR index Value."""
        if isinstance(node, ast.Constant):
            return arith.ConstantOp(IndexType.get(self.ctx), node.value)
        val = self.visit(node)
        if isinstance(val.type, IndexType):
            return val
        return arith.IndexCastOp(IndexType.get(self.ctx), val)

    def _build_index_or_range(self, node):
        """Convert AST node to (start_value, is_range) tuple.

        For slice syntax (start:end), returns (start_value, True).
        For index syntax (value), returns (value, False).
        """
        if isinstance(node, ast.Slice):
            if node.lower is None:
                self._raise_error(node, "Slice must have explicit start index")
            if node.upper is None:
                self._raise_error(node, "Slice must have explicit stop index")
            if node.step is not None:
                self._raise_error(node, "Slice step is not supported")
            start_val = self._to_index_value(node.lower)
            return (start_val, True)
        else:
            return (self._to_index_value(node), False)

    # Override to use i64 for all integer constants (attributes or not)
    # D2M ops require i64, and this reduces casts throughout the pipeline
    def visit_Constant(self, node):
        as_attr = getattr(node, "_ttkernel_as_attr", False)
        op_constructor = IntegerAttr.get if as_attr else arith.ConstantOp
        if callable(as_attr):
            return as_attr(node)
        elif isinstance(node.value, bool):
            return op_constructor(IntegerType.get_signless(1, self.ctx), node.value)
        elif isinstance(node.value, int):
            return op_constructor(IntegerType.get_signless(64, self.ctx), node.value)
        elif isinstance(node.value, str):
            return node.value
        else:
            self._raise_error(
                node, f"constant type {type(node.value).__name__} not implemented"
            )

    def visit_List(self, node):
        """Parse a list of constants. Returns a Python list, not MLIR values."""
        result = []
        for elt in node.elts:
            if not isinstance(elt, ast.Constant):
                self._raise_error(node, "list elements must be constants")
            result.append(elt.value)
        return result

    def _emit_cb_from_capture(self, cb):
        """Emit ttl.bind_cb for a captured CircularBuffer instance."""
        ttcore_dtype = TensorDtype(dtype=cb.dtype).ttcore_dtype
        element_type = ttcore.ir.TileType.get(
            self.ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
        )
        cb_type = ttl.CircularBufferType.get(
            self.ctx,
            list(cb.shape),
            element_type,
            cb.buffer_factor,
        )
        # Emit: %cb = ttl.bind_cb {cb_index = N, buffer_factor = M} : !ttl.cb<...>
        return ttl.bind_cb(cb_type, cb._cb_index, buffer_factor=cb.buffer_factor)

    def _validate_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Validate function has correct arguments for kernel entry."""
        assert not self.func_entry, "Cannot declare function within a function"
        if node.args.args:
            self._raise_error(
                node,
                "Thread functions must have no parameters. "
                "Use make_circular_buffer_like() in kernel body and capture CBs in closures.",
            )

    def _process_tensor_captures(
        self, captures: dict[str, object]
    ) -> list[tuple[str, object, object]]:
        """Process captured tensors and return (name, tensor, mlir_type) tuples."""
        processed: list[tuple[str, object, object]] = []
        for name, val in captures.items():
            is_tensor = is_ttnn_tensor(val)
            if not is_tensor:
                try:
                    import torch

                    is_tensor = isinstance(val, torch.Tensor)
                except ImportError:
                    pass
            if not is_tensor:
                continue

            tensor_type = _build_tensor_type(
                self.ctx,
                val,
                self.context.grid,
                self.context.tiled,
                self.context.memory_space,
            )
            processed.append((name, val, tensor_type))
        return processed

    def _setup_symbol_table(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, processed_captures: list[tuple[str, object, object]]
    ) -> Block:
        """Initialize symbol table with function arguments and captures."""
        self._tensor_accessor_names = [name for name, _val, _tt in processed_captures]
        self._tensor_accessor_global_indices = [
            get_tensor_global_index(val) for _name, val, _tt in processed_captures
        ]
        func_arg_types = [tt for _name, _val, tt in processed_captures]

        self.func_entry = func.FuncOp(name=node.name, type=(func_arg_types, []))

        # Set thread attribute: ttl.kernel_thread = #ttkernel.thread<compute/noc>
        thread_type = get_thread_type_string(self.kernel_type)
        thread_attr = ttkernel.ir.ThreadTypeAttr.get(self.ctx, thread_type)
        self.func_entry.attributes["ttl.kernel_thread"] = thread_attr

        self.symbol_tables.append({})
        func_bb: Block = self.func_entry.add_entry_block()

        # Add ttl module to symbol table
        self.symbol_tables[-1]["ttl"] = ttl

        # Ensure TTL dialect is registered for type parsing
        ttl.ensure_dialects_registered(self.ctx)

        self.module_symbol_table = SymbolTable(self.module.operation)
        return func_bb

    def _emit_entry(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self._validate_function_signature(node)

        processed_tensors = self._process_tensor_captures(self.captures)
        func_bb = self._setup_symbol_table(node, processed_tensors)

        # Emit function body
        with InsertionPoint(func_bb):
            # Map TensorAccessor function arguments to symbol table
            for i, name in enumerate(self._tensor_accessor_names):
                self.symbol_tables[-1][name] = func_bb.arguments[i]
                self.streams.add(name)

            # Prepopulate other captures (non-tensor)
            from ..circular_buffer import CircularBuffer

            for name, val in self.captures.items():
                if is_ttnn_tensor(val):
                    continue  # Already handled via function arguments
                # torch.Tensor in compile-only path: same as ttnn, handled via args
                try:
                    import torch

                    if isinstance(val, torch.Tensor):
                        continue
                except ImportError:
                    pass
                assert isinstance(name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
                elif isinstance(val, CircularBuffer):
                    cb_val = self._emit_cb_from_capture(val)
                    self.symbol_tables[-1][name] = cb_val
                else:
                    self._raise_error(
                        node, f"Invalid capture type for var {name}: {type(val)}"
                    )

            for target in node.body:
                self.visit(target)

            self._close_final_signpost()
            func.ReturnOp([])

        self.symbol_tables.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        with self._loc_for_node(node):
            return self._emit_entry(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        with self._loc_for_node(node):
            return self._emit_entry(node)

    def _get_cb_tensor_type(self, cb_val, node=None):
        """Extract the tensor type from a TTL CB type."""
        cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
        if cb_type is None:
            msg = f"Expected CircularBufferType, got {cb_val.type}"
            if node is not None:
                self._raise_error(node, msg)
            raise ValueError(msg)
        return RankedTensorType.get(cb_type.shape, cb_type.element_type)

    def visit_With(self, node):
        """
        Handle 'with' for CircularBuffer acquire/release.

        Acquire ops (wait/reserve) are generated left-to-right.
        Release ops (pop/push) are generated in reverse order at scope end.

        Example:
            with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
                ...
                # releases in reverse order: push(out), pop(rhs), pop(lhs)
        """
        with self._loc_for_node(node):
            releases: list[tuple[str, object, object, ast.AST]] = []

            for item in node.items:
                cb_val, method_name, expr_node, optional_name = self._parse_with_item(
                    item
                )
                acquire_result, release_info = self._emit_cb_acquire(
                    cb_val=cb_val,
                    method_name=method_name,
                    expr_node=expr_node,
                )
                releases.append(release_info)
                if optional_name is not None:
                    self.symbol_tables[-1][optional_name] = acquire_result

            for stmt in node.body:
                self.visit(stmt)

            self._emit_cb_releases(releases)

    def _parse_with_item(
        self, item: ast.withitem
    ) -> tuple[object, str, ast.AST, str | None]:
        """Parse a single with-item and return (cb_value, method_name, expr_node, optional_name)."""
        context_expr = item.context_expr
        optional_vars = item.optional_vars

        if not isinstance(context_expr, ast.Call):
            self._raise_error(
                context_expr,
                "'with' requires a method call (e.g., cb.reserve())",
            )

        if not isinstance(context_expr.func, ast.Attribute):
            self._raise_error(
                context_expr, "'with' requires a method call on an object"
            )

        method_name = context_expr.func.attr
        cb_node = context_expr.func.value

        if method_name not in ("reserve", "wait"):
            self._raise_error(
                context_expr,
                f"'with' only supports 'reserve()' or 'wait()', got '{method_name}'",
            )

        if not isinstance(cb_node, ast.Name):
            self._raise_error(
                context_expr,
                "'with' requires a simple variable (e.g., cb.reserve())",
            )

        cb_table = self._var_exists(cb_node.id)
        if not cb_table:
            self._raise_error(cb_node, f"'{cb_node.id}' not found in scope")
        cb_val = cb_table[cb_node.id]

        optional_name: str | None = None
        if optional_vars is not None:
            if not isinstance(optional_vars, ast.Name):
                self._raise_error(
                    optional_vars, "'with ... as var' requires a simple variable name"
                )
            optional_name = optional_vars.id

        return cb_val, method_name, context_expr, optional_name

    def _emit_cb_acquire(
        self, *, cb_val: object, method_name: str, expr_node: ast.AST
    ) -> tuple[object, tuple[str, object, object, ast.AST]]:
        """Emit CB acquire operation with signposts if profiling enabled."""
        tensor_type = self._get_cb_tensor_type(cb_val, node=expr_node)
        if method_name == "reserve":
            tensor = self._emit_op_signposts(
                "cb_reserve",
                expr_node,
                lambda tt=tensor_type, cv=cb_val: ttl.cb_reserve(tt, cv),
            )
            release_info = ("cb_push", ttl.cb_push, cb_val, expr_node)
        else:
            tensor = self._emit_op_signposts(
                "cb_wait",
                expr_node,
                lambda tt=tensor_type, cv=cb_val: ttl.cb_wait(tt, cv),
            )
            release_info = ("cb_pop", ttl.cb_pop, cb_val, expr_node)

        acquire_result = ttl.attach_cb(tensor.type, tensor, cb_val)
        return acquire_result, release_info

    def _emit_cb_releases(
        self, releases: list[tuple[str, object, object, ast.AST]]
    ) -> None:
        """Emit CB release operations for all acquired CBs."""
        for op_name, release_op, cb_val, expr_node in reversed(releases):
            self._emit_op_signposts(
                op_name,
                expr_node,
                lambda ro=release_op, cv=cb_val: ro(cv),
                implicit=True,
            )


def syntax(syntax_name):
    if syntax_name.startswith("!"):

        def _class_wrapper(cls):
            assert isinstance(cls, type)

            for name, method in cls.__dict__.items():
                if callable(method):
                    sig = inspect.signature(method)
                    first_arg_name = next(iter(sig.parameters.keys()))
                    if first_arg_name == "ast_self":
                        setattr(cls, name, staticmethod(method))
                        qualified = f"{syntax_name}.{name}"
                        TTLGenericCompiler._syntax[qualified] = method

            return cls

        return _class_wrapper
    else:

        def _fn_wrapper(fn):
            assert callable(fn)
            TTLGenericCompiler._syntax[fn.__name__] = fn
            return fn

        return _fn_wrapper
