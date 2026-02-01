# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Thread registry for automatic collection of @compute and @datamovement threads."""

from __future__ import annotations

from collections.abc import Callable


class ThreadRegistry:
    """Registry for automatic collection of @compute and @datamovement threads."""

    __slots__ = ("_threads",)

    def __init__(self) -> None:
        self._threads: list[Callable[..., object]] = []

    def register(self, thread_fn: Callable[..., object]) -> None:
        """Register a thread function during decoration."""
        self._threads.append(thread_fn)

    def clear(self) -> None:
        """Clear the registry before kernel execution."""
        self._threads.clear()

    def get_and_clear(self) -> list[Callable[..., object]]:
        """Return all registered threads and clear the registry."""
        threads = list(self._threads)
        self._threads.clear()
        return threads


_thread_registry = ThreadRegistry()


def get_thread_registry() -> ThreadRegistry:
    """Return the global thread registry used by compile pipeline and decorators."""
    return _thread_registry
