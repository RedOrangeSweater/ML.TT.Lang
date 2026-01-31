# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Loader and load_and_patch: resolve (module, symbol) to a callable and patch the slot.

Default strategy: load a module from a path (file or package) and take the callable
by symbol name. Can be replaced or extended for network/cache.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .core import LazyLoadRequired, LazyWrapper


def load_callable_from_module_path(
    module_path: Path,
    symbol: str,
    module_name: str | None = None,
) -> Callable[..., Any]:
    """Load a module from a file path and return the callable named by symbol."""
    if module_name is None:
        module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    obj = getattr(mod, symbol)
    if not callable(obj):
        raise TypeError(f"{symbol!r} in {module_path} is not callable")
    return obj


def load_callable_from_dotted_path(
    dotted_module: str,
    symbol: str,
) -> Callable[..., Any]:
    """Load a callable from an already-importable dotted module path."""
    mod = __import__(dotted_module, fromlist=[symbol])
    obj = getattr(mod, symbol)
    if not callable(obj):
        raise TypeError(f"{symbol!r} in {dotted_module} is not callable")
    return obj


LoaderFn = Callable[[ModuleType, str], Callable[..., Any] | None]


def make_dir_resolver(implementations_dir: Path) -> LoaderFn:
    """Resolve (module, symbol) to a callable by loading implementations_dir / {symbol}.py."""
    implementations_dir = Path(implementations_dir)

    def symbol_to_path(_module: ModuleType, symbol: str) -> Path | None:
        path = implementations_dir / f"{symbol}.py"
        return path if path.exists() else None

    return make_file_resolver(symbol_to_path)


def make_load_and_patch(
    resolve: Callable[[ModuleType, str], Callable[..., Any] | None],
) -> Callable[[LazyLoadRequired], None]:
    """Build load_and_patch that uses resolve(module, symbol) -> callable or None.

    If resolve returns None, the exception is re-raised. Otherwise the callable
    is installed: if the slot is a LazyWrapper, set wrapper._impl; else set module[symbol].
    """

    def load_and_patch(e: LazyLoadRequired) -> None:
        impl = resolve(e.module, e.symbol)
        if impl is None:
            raise e
        current = getattr(e.module, e.symbol)
        if isinstance(current, LazyWrapper):
            current._impl = impl
        else:
            setattr(e.module, e.symbol, impl)

    return load_and_patch


def make_file_resolver(
    symbol_to_path: Callable[[ModuleType, str], Path | None],
) -> LoaderFn:
    """Build a resolver that loads from a file path returned by symbol_to_path."""

    def resolve(module: ModuleType, symbol: str) -> Callable[..., Any] | None:
        path = symbol_to_path(module, symbol)
        if path is None:
            return None
        return load_callable_from_module_path(path, symbol)

    return resolve
