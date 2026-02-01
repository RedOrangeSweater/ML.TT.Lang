# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Callback protocol and lifecycle events for ttl.nn.Trainer.

All methods are optional; override only the hooks you need. Event names align with
PyTorch Lightning-style lifecycle (on_fit_start, on_train_batch_end, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Callback(Protocol):
    """Protocol for Trainer callbacks. All methods optional; default no-op."""

    def on_fit_start(self, trainer: Any) -> None:
        """Called when fit starts (before compile warmup)."""
        ...

    def on_fit_end(self, trainer: Any) -> None:
        """Called when fit ends (after run loop)."""
        ...

    def on_compile_start(self, trainer: Any) -> None:
        """Called before compile warmup."""
        ...

    def on_compile_end(self, trainer: Any) -> None:
        """Called after compile warmup."""
        ...

    def on_train_batch_start(
        self, trainer: Any, batch: Any, batch_idx: int
    ) -> None:
        """Called before a train batch (or benchmark step)."""
        ...

    def on_train_batch_end(
        self, trainer: Any, batch: Any, batch_idx: int, result: Any
    ) -> None:
        """Called after a train batch (or benchmark step). result is step output."""
        ...

    def on_validation_batch_end(
        self, trainer: Any, batch: Any, batch_idx: int, result: Any
    ) -> None:
        """Called after a validation batch."""
        ...

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at epoch start."""
        ...

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Called at epoch end."""
        ...


def _noop(*args: Any, **kwargs: Any) -> None:
    pass


class CallbackAdapter:
    """Base adapter: implements all Callback methods as no-ops. Override only what you need."""

    def on_fit_start(self, trainer: Any) -> None:
        _noop(trainer)

    def on_fit_end(self, trainer: Any) -> None:
        _noop(trainer)

    def on_compile_start(self, trainer: Any) -> None:
        _noop(trainer)

    def on_compile_end(self, trainer: Any) -> None:
        _noop(trainer)

    def on_train_batch_start(
        self, trainer: Any, batch: Any, batch_idx: int
    ) -> None:
        _noop(trainer, batch, batch_idx)

    def on_train_batch_end(
        self, trainer: Any, batch: Any, batch_idx: int, result: Any
    ) -> None:
        _noop(trainer, batch, batch_idx, result)

    def on_validation_batch_end(
        self, trainer: Any, batch: Any, batch_idx: int, result: Any
    ) -> None:
        _noop(trainer, batch, batch_idx, result)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        _noop(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        _noop(trainer, epoch)
