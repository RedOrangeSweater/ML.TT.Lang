# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Context manager and helper for verbose compilation output.

Use verbose_compilation(enable) as a context manager and verbose_print(...)
inside it so that output is emitted only when verbose is enabled. Avoids
scattered `if verbose: print(...)` across the codebase.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_verbose_compilation: ContextVar[bool] = ContextVar(
    "verbose_compilation", default=False
)


@contextmanager
def verbose_compilation(enable: bool):
    """Context manager that sets verbose compilation flag for the current context.

    Inside the block, verbose_print() will emit output only when enable is True.
    """
    token = _verbose_compilation.set(enable)
    try:
        yield
    finally:
        _verbose_compilation.reset(token)


def verbose_print(*args: object, **kwargs: object) -> None:
    """Print only when inside a verbose_compilation(True) context."""
    if _verbose_compilation.get():
        print(*args, **kwargs)
