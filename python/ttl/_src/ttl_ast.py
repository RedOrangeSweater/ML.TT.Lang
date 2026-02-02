# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
from collections.abc import Callable, Generator
from typing import Any, NoReturn, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from ttmlir.dialects import arith, func, ttcore, ttkernel
from ttmlir.ir import (
    Block,
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
from .ast_proxies import (
    AssignProxy,
    AttributeProxy,
    BinOpProxy,
    CallProxy,
    NodeProxy,
    SubscriptProxy,
)
from .auto_profile import (
    Signpost,
    SignpostBoundary,
    get_line_mapper,
    is_auto_profile_enabled,
)
from .context import (
    CompilerContext,
    ThreadSourceInfo,
    TTLCompilerConfig,
)
from .ir_builder import IRBuilder
from .profiling_aspect import ProfilingAspect
from .semantic_dispatcher import SemanticDispatcher
from .semantic_proxies import WithCBProxy
from .tensor_registry import get_tensor_global_index
from .type_builder import build_tensor_type as _build_tensor_type

_T = TypeVar("_T")
_P = TypeVar("_P", bound=NodeProxy)


def with_proxy(proxy_cls: type[_P]):
    """Decorator to wrap the first argument (node) into a proxy and handle common boilerplate."""

    def decorator(method: Callable[[Any, _P], Any]):
        @functools.wraps(method)
        def wrapper(self: TTLGenericCompiler, node: ast.AST, *args, **kwargs):
            # 1. Semantic dispatch
            result = self.dispatcher.dispatch(node)
            if result is not None:
                return result

            # 2. Wrap in proxy
            proxy = proxy_cls(node)

            # 3. Set location context and handle profiling/errors
            with self._loc_for_node(node):
                try:
                    return self.profiler.wrap_visit(node, lambda: method(self, proxy, *args, **kwargs))
                except (ValueError, TypeError, NotImplementedError) as e:
                    if isinstance(e, TTLangCompileError):
                        raise
                    proxy.error(str(e), self.config.source_file, self.config.line_offset)

        return wrapper

    return decorator


class TTLGenericCompiler(TTCompilerBase, BaseModel):
    """Compiler that generates TTL dialect ops from Python AST."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _syntax: dict[str, Callable] = {}

    # Pydantic Fields
    name: str
    kernel_type: str | None = None
    config: TTLCompilerConfig
    captures: dict[str, Any] = Field(default_factory=dict)
    args: tuple[Any, ...] = Field(default_factory=tuple)
    init_kwargs: dict[str, Any] = Field(default_factory=dict, alias="kwargs")

    # State & Components
    streams: set[str] = Field(default_factory=set)
    auto_profile_enabled: bool = Field(default_factory=is_auto_profile_enabled)
    line_mapper: Any = Field(default=None)
    loc: Location = Field(init=False)
    context: CompilerContext = Field(init=False)
    profiler: ProfilingAspect = Field(init=False)
    ir: IRBuilder = Field(init=False)
    dispatcher: SemanticDispatcher = Field(init=False)

    # Private Attributes
    _current_line_signpost: Signpost | None = PrivateAttr(default=None)
    _cb_info: list[dict[str, object]] = PrivateAttr(default_factory=list)
    _fn_map: dict[str, Callable] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _init_compiler(self) -> Self:
        """Initialize both base classes and internal components."""
        # 1. Initialize TTCompilerBase (non-Pydantic)
        TTCompilerBase.__init__(self, self.name, self.kernel_type, *self.args, **self.init_kwargs)

        # 2. Setup TTL-specific internals
        self.loc = Location.name(self.name)
        self.supported_nodes.extend([ast.AsyncFunctionDef, ast.With])

        self.context = CompilerContext(
            grid=self.config.grid,
            memory_space=self.config.memory_space,
            tiled=self.config.tiled,
        )

        if self.auto_profile_enabled:
            self.line_mapper = get_line_mapper()
            if self.line_mapper:
                self.line_mapper.line_offset = self.config.line_offset

        self.profiler = ProfilingAspect(self)
        self.ir = IRBuilder(self)
        self.dispatcher = SemanticDispatcher(self)
        self._fn_map = dict(TTLGenericCompiler._syntax)

        return self

    def __init__(
        self,
        name: str,
        kernel_type: str | None = None,
        captures: dict[str, Any] | None = None,
        *args: Any,
        config: TTLCompilerConfig | None = None,
        **kwargs: Any,
    ):
        # Prepare config if not provided
        if config is None:
            config = TTLCompilerConfig.model_validate(kwargs)

        # Initialize via BaseModel, which triggers _init_compiler validator
        BaseModel.__init__(
            self,
            name=name,
            kernel_type=kernel_type,
            config=config,
            captures=captures or {},
            args=args,
            init_kwargs=kwargs,
        )

    @property
    def source_info(self) -> ThreadSourceInfo:
        """Structured source info for error reporting and profiling."""
        return ThreadSourceInfo(
            source_file=self.config.source_file,
            source_lines=self.config.source_lines,
            line_offset=self.config.line_offset,
        )

    @with_proxy(AssignProxy)
    @contextlib.contextmanager
    def scope(self) -> Generator[None, None, None]:
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

    def visit_Assign(self, proxy: AssignProxy):
        """Handle tuple unpacking for TTL functions like core(dims=2)."""
        target = proxy.first_target
        if not isinstance(target, ast.Tuple):
            return super().visit_Assign(proxy.node)

        value = self.visit(proxy.value)
        if not isinstance(value, tuple):
            return super().visit_Assign(proxy.node)

        targets = target.elts
        if len(value) != len(targets):
            raise ValueError(
                f"Cannot unpack {len(value)} values into {len(targets)} variables"
            )

        for elt, val in zip(targets, value, strict=False):
            if not isinstance(elt, ast.Name):
                raise ValueError("Tuple unpacking requires simple variable names")
            self.define(elt.id, val)

    def _loc_for_node(self, node: ast.AST) -> Location:
        """Return file location for node if debug_locations enabled, else name location."""
        if self.config.debug_locations:
            proxy = NodeProxy(node)
            if proxy.lineno is not None:
                return proxy.location(self.ctx, self.config.source_file, self.config.line_offset)
        return self.loc

    def _raise_error(self, node: ast.AST, message: str) -> NoReturn:
        """Raise a TTLangCompileError with source location from AST node."""
        NodeProxy(node).error(message, self.config.source_file, self.config.line_offset)

    # Auto-profiling helpers for line-based signposting

    def _emit_marker(self, signpost: Signpost, boundary: SignpostBoundary) -> None:
        """Emit a signpost marker (before or after) for a typed signpost."""
        ttl.signpost(signpost.get_marker(boundary))  # type: ignore[attr-defined]

    def _register_pair(self, signpost: Signpost, source_line: str) -> None:
        """Register before/after signpost pair if line_mapper is active."""
        if not self.line_mapper:
            return

        for marker in (signpost.before(), signpost.after()):
            self.line_mapper.register_signpost(marker, signpost.file_lineno, source_line)

    def _emit_line_signpost_if_needed(self, node: ast.AST) -> None:
        """Emit signposts at line boundaries for auto-profiling."""
        proxy = NodeProxy(node)
        if not self.auto_profile_enabled or not isinstance(proxy.lineno, int):
            return

        signpost = proxy.line_signpost(self.config.line_offset)
        if self._current_line_signpost and self._current_line_signpost.file_lineno == signpost.file_lineno:
            return

        if self._current_line_signpost:
            self._emit_marker(self._current_line_signpost, SignpostBoundary.AFTER)

        self._register_pair(signpost, proxy.source_line(self.config.source_lines, self.config.line_offset))
        self._emit_marker(signpost, SignpostBoundary.BEFORE)
        self._current_line_signpost = signpost

    def _close_final_signpost(self):
        """Close the final signpost at the end of function body."""
        if self.auto_profile_enabled and self._current_line_signpost is not None:
            self._emit_marker(self._current_line_signpost, SignpostBoundary.AFTER)
            self._current_line_signpost = None

    def _try_emit_auto_signposts(self, node: ast.AST, visit_fn: Callable[[], _T]) -> _T:
        """Emit line-based signposts if auto-profiling is enabled."""
        self._emit_line_signpost_if_needed(node)
        return visit_fn()

    @contextlib.contextmanager
    def signpost_context(
        self,
        node: ast.AST,
        signpost: Signpost | None = None,
    ) -> Generator[None]:
        """Context manager to emit signposts around an operation."""
        if not self.auto_profile_enabled or signpost is None or not signpost.is_valid:
            with self._loc_for_node(node):
                yield
            return

        self._register_pair(signpost, NodeProxy(node).source_line(self.config.source_lines, self.config.line_offset))

        with self._loc_for_node(node):
            self._emit_marker(signpost, SignpostBoundary.BEFORE)
            yield
            self._emit_marker(signpost, SignpostBoundary.AFTER)

    def _emit_op_signposts(
        self,
        op_name: str,
        node: ast.AST,
        op_fn: Callable[[], _T],
        implicit: bool = False,
    ) -> _T:
        """Emit signposts for operations with op name included."""
        with self.profiler.instrument_op(node, op_name, implicit):
            return op_fn()

    @with_proxy(CallProxy)
    def visit_Call(self, proxy: CallProxy):
        """Override to set location context, catch errors, and inject auto-profiling."""
        return super().visit_Call(proxy.node)

    @with_proxy(BinOpProxy)
    def visit_BinOp(self, proxy: BinOpProxy):
        """Override to inject auto-profiling and provide better error messages."""
        return super().visit_BinOp(proxy.node)

    def visit_Name(self, node):
        """Override to check function globals for simple constants."""
        result = super().visit_Name(node)
        if result is not None:
            return result

        # Check if it's a module-level constant
        var_name = node.id
        if var_name in self.config.fn_globals:
            val = self.config.fn_globals[var_name]
            if isinstance(val, int):
                return self.ir.const_i64(val)
            if isinstance(val, float):
                return self.ir.const_f32(val)

        return None

    def visit_children(self, node: ast.AST) -> list[Any]:
        """Visit all children of a node and return results."""
        return [self.visit(child) for child in ast.iter_child_nodes(node)]

    @with_proxy(AttributeProxy)
    def visit_Attribute(
        self,
        proxy: AttributeProxy,
        func_args: list[object] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> object | None:
        """Override to set location context and catch errors for method calls."""
        func_args = [] if func_args is None else func_args
        kwargs = {} if kwargs is None else kwargs

        # Handle ttl.XXX and ttl.math.XXX attribute access
        if proxy.is_ttl_module or proxy.is_ttl_math:
            namespace = "ttl" if proxy.is_ttl_module else "ttl.math"
            if fn := self._fn_map.get(proxy.attr):
                return fn(*func_args, **kwargs)
            proxy.error(f"Unknown function: {namespace}.{proxy.attr}", self.config.source_file, self.config.line_offset)

        return super().visit_Attribute(proxy.node, func_args, kwargs)

    @with_proxy(SubscriptProxy)
    def visit_Subscript(self, proxy: SubscriptProxy):
        """Handle tensor[row, col] or tensor[r0:r1, c0:c1] indexing."""
        value_proxy = NodeProxy(proxy.value)
        if not value_proxy.is_name:
            proxy.error("TTL only supports subscripting simple variables", self.config.source_file, self.config.line_offset)

        var_name = value_proxy.name_id
        assert var_name is not None
        tbl = self._var_exists(var_name)
        if not tbl:
            proxy.error(f"Unknown variable: {var_name}", self.config.source_file, self.config.line_offset)

        tensor = tbl[var_name]
        if not isinstance(getattr(tensor, "type", None), RankedTensorType):
            proxy.error("TTL only supports subscripting tensors", self.config.source_file, self.config.line_offset)

        if isinstance(proxy.slice, ast.Tuple):
            indices = [self._build_index_or_range(elt) for elt in proxy.slice.elts]
        else:
            indices = [self._build_index_or_range(proxy.slice)]

        return (tensor, indices)

    def _to_index_value(self, node):
        """Convert AST node to MLIR index Value."""
        if isinstance(node, ast.Constant):
            return self.ir.const_index(node.value)
        val = self.visit(node)
        if isinstance(val.type, IndexType):
            return val
        return self.ir.index_cast(val)

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
        if callable(as_attr):
            return as_attr(node)

        if isinstance(node.value, bool):
            type_ = IntegerType.get_signless(1, self.ctx)
            return IntegerAttr.get(type_, node.value) if as_attr else arith.ConstantOp(type_, node.value).result
        elif isinstance(node.value, int):
            return IntegerAttr.get(IntegerType.get_signless(64, self.ctx), node.value) if as_attr else self.ir.const_i64(node.value)
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

        # Push global scope
        self.symbol_tables.append({})
        func_bb: Block = self.func_entry.add_entry_block()

        # Add ttl module to symbol table
        self.define("ttl", ttl)

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
                self.define(name, func_bb.arguments[i])
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
                    self.define(name, arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    ))
                elif isinstance(val, CircularBuffer):
                    cb_val = self._emit_cb_from_capture(val)
                    self.define(name, cb_val)
                else:
                    self._raise_error(
                        node, f"Invalid capture type for var {name}: {type(val)}"
                    )

            for target in node.body:
                self.visit(target)

            self.profiler.close_final_signpost()
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
        """
        with self._loc_for_node(node):
            releases: list[tuple[str, object, object, ast.AST]] = []

            for item in node.items:
                proxy = WithCBProxy(item)

                if not proxy.is_valid_cb_method:
                    proxy.error("'with' only supports 'reserve()' or 'wait()' on CircularBuffer", self.config.source_file, self.config.line_offset)

                cb_table = self._var_exists(proxy.cb_var_name)
                if not cb_table:
                    proxy.error(f"'{proxy.cb_var_name}' not found in scope", self.config.source_file, self.config.line_offset)
                cb_val = cb_table[proxy.cb_var_name]

                acquire_result, release_info = self._emit_cb_acquire(
                    cb_val=cb_val,
                    method_name=proxy.method_name,
                    expr_node=item.context_expr,
                )
                releases.append(release_info)
                if proxy.optional_name is not None:
                    self.define(proxy.optional_name, acquire_result)

            for stmt in node.body:
                self.visit(stmt)

            self._emit_cb_releases(releases)

    def _emit_cb_acquire(
        self, *, cb_val: object, method_name: str, expr_node: ast.AST
    ) -> tuple[object, tuple[str, object, object, ast.AST]]:
        """Emit CB acquire operation with signposts if profiling enabled."""
        tensor_type = self._get_cb_tensor_type(cb_val, node=expr_node)
        if method_name == "reserve":
            tensor = self._emit_op_signposts(
                "cb_reserve",
                expr_node,
                lambda tt=tensor_type, cv=cb_val: ttl.cb_reserve(tt, cv),  # type: ignore[attr-defined, misc]
            )
            release_info = ("cb_push", ttl.cb_push, cb_val, expr_node)  # type: ignore[attr-defined]
        else:
            tensor = self._emit_op_signposts(
                "cb_wait",
                expr_node,
                lambda tt=tensor_type, cv=cb_val: ttl.cb_wait(tt, cv),  # type: ignore[attr-defined, misc]
            )
            release_info = ("cb_pop", ttl.cb_pop, cb_val, expr_node)  # type: ignore[attr-defined]

        acquire_result = ttl.attach_cb(tensor.type, tensor, cb_val)  # type: ignore[attr-defined]
        return acquire_result, release_info

    def _emit_cb_releases(
        self, releases: list[tuple[str, object, object, ast.AST]]
    ) -> None:
        """Emit CB release operations for all acquired CBs."""
        for op_name, release_op, cb_val, expr_node in reversed(releases):
            self._emit_op_signposts(
                op_name,
                expr_node,
                lambda ro=release_op, cv=cb_val: ro(cv),  # type: ignore[misc]
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
