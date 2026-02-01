# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile thread decorator: @compute and @datamovement implementation."""

from __future__ import annotations

import ast
import functools
import inspect
from collections.abc import Callable

from .._src.ttl_ast import TTLCompilerConfig, TTLGenericCompiler
from ..diagnostics import format_mlir_error
from ..verbose_context import verbose_compilation, verbose_print
from .registry import get_thread_registry
from .source_context import CompilationSourceContext, collect_captures


def _get_compilation_context(
    f: Callable[..., object],
    verbose: bool,
    **kwargs: object,
) -> tuple[CompilationSourceContext, TTLCompilerConfig]:
    """Build source context and compiler config for a thread function."""
    ctx = CompilationSourceContext.from_function(f, verbose=verbose)
    config = TTLCompilerConfig(source_context=ctx, **kwargs)
    return ctx, config


def _compile_thread_core(
    f: Callable[..., object],
    ctx: CompilationSourceContext,
    config: TTLCompilerConfig,
    kernel_type: str,
    *args: object,
) -> TTLGenericCompiler:
    """Parse AST, compile to MLIR, verify; return compiler result."""
    m = ast.parse(ctx.source_code)
    b = TTLGenericCompiler(
        f.__name__,
        kernel_type,
        collect_captures(f),
        *args,
        config=config,
    )
    with verbose_compilation(ctx.verbose):
        verbose_print(ast.dump(m, indent=4) + "\n")
        b.visit(m)
        verbose_print(b.module)
    try:
        b.module.operation.verify()
    except Exception as e:
        formatted = format_mlir_error(
            str(e), ctx.source_lines, ctx.source_file
        )
        raise RuntimeError(formatted) from None
    return b


def compile_thread(
    f: Callable[..., object],
    kernel_type: str,
    verbose: bool = False,
) -> Callable[..., object]:
    """
    Compile a kernel thread function (compute or datamovement).

    Returns a wrapper that compiles the function body to MLIR and registers
    the thread for collection by the compile pipeline.
    """
    try:
        source_file = inspect.getfile(f)
    except (TypeError, OSError):
        source_file = "<unknown>"

    @functools.wraps(f)
    def _wrapper(*args: object, **kwargs: object) -> object:
        ctx, config = _get_compilation_context(f, verbose, **kwargs)
        return _compile_thread_core(f, ctx, config, kernel_type, *args)

    _wrapper._decorator_name = kernel_type + "_thread"  # type: ignore[attr-defined]
    _wrapper._source_file = source_file  # type: ignore[attr-defined]
    get_thread_registry().register(_wrapper)

    if inspect.ismethod(f):
        return staticmethod(_wrapper)
    return _wrapper
