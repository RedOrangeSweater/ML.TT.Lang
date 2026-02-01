# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-style Module base, TrainableModule (Lightning-like), and ProgramModule adapter for ttl.run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..program import ProgramOptions, ProgramSpec
from .device import TensorAdapter


@runtime_checkable
class Module(Protocol):
    """Protocol for runnable modules: single __call__ with tensors, returns result."""

    def __call__(self, *args: object, **kwargs: object) -> object | None:
        """Run the module with given tensors. Returns output tensor or None (compile-only)."""
        ...


@runtime_checkable
class TrainableModule(Protocol):
    """Lightning-like protocol: forward, training_step, validation_step, configure_optimizers (optional stubs)."""

    def forward(self, *args: object, **kwargs: object) -> object | None:
        """Forward pass (inference). Default: delegate to __call__ or training_step without grad."""
        ...

    def training_step(self, batch: Any, batch_idx: int) -> object | None:
        """One training (or benchmark) step. Returns step output (e.g. loss or None)."""
        ...

    def validation_step(self, batch: Any, batch_idx: int) -> object | None:
        """One validation step. Optional; default no-op."""
        ...

    def configure_optimizers(self) -> Any:
        """Return optimizer(s) and/or scheduler(s). Optional stub; default None."""
        ...


class TrainableModuleAdapter:
    """Base adapter for TrainableModule: stubs for training_step, validation_step, configure_optimizers.

    Subclass and override forward (and optionally training_step) for use with Trainer.
    """

    def forward(self, *args: object, **kwargs: object) -> object | None:
        """Override: inference pass. Default returns None."""
        return None

    def training_step(self, batch: Any, batch_idx: int) -> object | None:
        """Override: one training/benchmark step. Default returns None."""
        return None

    def validation_step(self, batch: Any, batch_idx: int) -> object | None:
        """Override: one validation step. Default returns None."""
        return None

    def configure_optimizers(self) -> Any:
        """Override: return optimizer/scheduler. Default None (inference/benchmark only)."""
        return None


class ProgramModule:
    """Concrete module that wraps a @ttl.program callable and runs it via ttl.run.

    Optional tensor_adapter: when set, converts torch inputs to device (ttnn) and
    result back to host (torch). compile() runs one forward pass to warmup the compile cache.
    """

    def __init__(
        self,
        program: Callable[..., object],
        grid: tuple[int, ...] | list[int] | Callable[..., object],
        options: ProgramOptions | None = None,
        tensor_adapter: TensorAdapter | None = None,
    ) -> None:
        """Wrap a program and grid for use as a Module.

        Args:
            program: @ttl.program-decorated callable.
            grid: Grid dimensions or callable to resolve from args.
            options: Optional program options; default ProgramOptions().
            tensor_adapter: Optional adapter for torch<->ttnn conversion; when set, __call__ converts I/O.
        """
        self._program = program
        self._grid = grid
        self._options = options
        self._tensor_adapter = tensor_adapter

    def compile(self, *args: object, **kwargs: object) -> None:
        """Run one forward pass to warmup the compile cache. Separate from __call__ for explicit warmup."""
        self(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> object | None:
        """Run the program with given args via ttl.run. Converts I/O when tensor_adapter is set."""
        from ..program import RunRequest
        from ..ttl_api import run

        run_args: tuple[object, ...] = args
        run_kwargs: dict[str, object] = dict(kwargs)
        if self._tensor_adapter is not None:
            run_args = self._tensor_adapter.adapt_args(*args)
        opts = self._options if self._options is not None else ProgramOptions(num_outs=1)
        spec = ProgramSpec(program=self._program, grid=self._grid, options=opts)
        req = RunRequest(spec=spec, args=run_args, kwargs=run_kwargs)
        result = run(req)
        if self._tensor_adapter is not None and result is not None:
            return self._tensor_adapter.adapt_result(result)
        return result
