# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Core types for lazy boot: exception, placeholder factory, and trampoline wrapper.

Boot code hits a placeholder -> exception -> handler loads and patches -> retry
from the start of the augmented function with the same arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any


class LazyLoadRequired(Exception):
    """Raised by a placeholder when the real implementation must be loaded.

    Attributes:
        module: Module where the symbol lives (slot to patch).
        symbol: Name of the callable to replace.
        args: Positional arguments of the call (for retry).
        kwargs: Keyword arguments of the call (for retry).
    """

    def __init__(
        self,
        module: ModuleType,
        symbol: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super().__init__(module, symbol, args, kwargs)
        self.module = module
        self.symbol = symbol
        self.args = args
        self.kwargs = kwargs


def make_placeholder(module: ModuleType, symbol: str) -> Callable[..., None]:
    """Build a placeholder that raises LazyLoadRequired with (module, symbol, args, kwargs)."""

    def placeholder(*args: Any, **kwargs: Any) -> None:
        raise LazyLoadRequired(module, symbol, args, kwargs)

    return placeholder


class LazyWrapper:
    """Trampoline: calls current impl; on LazyLoadRequired, runs load_and_patch then retries.

    The slot (module[symbol]) can stay as this wrapper; the wrapper's _impl is updated
    from placeholder to the loaded callable. That allows later "unload" by resetting
    _impl back to a placeholder.
    """

    def __init__(
        self,
        module: ModuleType,
        symbol: str,
        placeholder: Callable[..., Any],
        load_and_patch: Callable[[LazyLoadRequired], None],
    ) -> None:
        self._module = module
        self._symbol = symbol
        self._impl = placeholder
        self._load_and_patch = load_and_patch

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._impl(*args, **kwargs)
        except LazyLoadRequired as e:
            self._load_and_patch(e)
            impl = getattr(self._module, self._symbol)
            return impl(*e.args, **e.kwargs)


def install_lazy_slot(
    module: ModuleType,
    symbol: str,
    load_and_patch: Callable[[LazyLoadRequired], None],
) -> None:
    """Set module[symbol] to a LazyWrapper with a placeholder for that symbol."""
    placeholder = make_placeholder(module, symbol)
    wrapper = LazyWrapper(module, symbol, placeholder, load_and_patch)
    setattr(module, symbol, wrapper)
