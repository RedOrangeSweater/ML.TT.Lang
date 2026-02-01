# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-style Module base and ProgramModule adapter for ttl.run."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from ..program import ProgramOptions, ProgramSpec


@runtime_checkable
class Module(Protocol):
    """Protocol for runnable modules: single __call__ with tensors, returns result."""

    def __call__(self, *args: object, **kwargs: object) -> object | None:
        """Run the module with given tensors. Returns output tensor or None (compile-only)."""
        ...


class ProgramModule:
    """Concrete module that wraps a @ttl.program callable and runs it via ttl.run."""

    def __init__(
        self,
        program: Callable[..., object],
        grid: tuple[int, ...] | list[int] | Callable[..., object],
        options: ProgramOptions | None = None,
    ) -> None:
        """Wrap a program and grid for use as a Module.

        Args:
            program: @ttl.program-decorated callable.
            grid: Grid dimensions or callable to resolve from args.
            options: Optional program options; default ProgramOptions().
        """
        self._program = program
        self._grid = grid
        self._options = options

    def __call__(self, *args: object, **kwargs: object) -> object | None:
        """Run the program with given args via ttl.run."""
        from ..program import RunRequest
        from ..ttl_api import run

        opts = self._options if self._options is not None else ProgramOptions(num_outs=1)
        spec = ProgramSpec(program=self._program, grid=self._grid, options=opts)
        req = RunRequest(spec=spec, args=args, kwargs=kwargs)
        return run(req)
