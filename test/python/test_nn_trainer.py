# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest: ttl.nn.Trainer lifecycle hooks and fit modes.

Tests callbacks (on_fit_start, on_compile_start/end, on_train_batch_*, on_epoch_*)
and fit() in benchmark vs scheduler_env mode.
"""

from __future__ import annotations

import os

import pytest

import ttl.nn


@pytest.fixture(autouse=True)
def compile_only() -> None:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"


class TestTrainerLifecycleHooks:
    """Trainer invokes callbacks in correct order."""

    def test_on_fit_start_end_and_compile_hooks(self) -> None:
        order: list[str] = []

        class HookTracker(ttl.nn.CallbackAdapter):
            def on_fit_start(self, trainer: ttl.nn.Trainer) -> None:
                order.append("on_fit_start")

            def on_compile_start(self, trainer: ttl.nn.Trainer) -> None:
                order.append("on_compile_start")

            def on_compile_end(self, trainer: ttl.nn.Trainer) -> None:
                order.append("on_compile_end")

            def on_fit_end(self, trainer: ttl.nn.Trainer) -> None:
                order.append("on_fit_end")

        config = ttl.nn.TrainerConfig(
            mode="benchmark",
            max_steps=1,
            compile_warmup_steps=0,
        )
        trainer = ttl.nn.Trainer(
            model=None,
            config=config,
            callbacks=[HookTracker()],
        )
        trainer.fit(train_batches=[None])
        assert order == [
            "on_fit_start",
            "on_compile_start",
            "on_compile_end",
            "on_fit_end",
        ]

    def test_on_epoch_and_batch_hooks(self) -> None:
        batch_ends: list[tuple[int, int]] = []

        class BatchTracker(ttl.nn.CallbackAdapter):
            def on_epoch_start(self, trainer: ttl.nn.Trainer, epoch: int) -> None:
                batch_ends.append((-1, epoch))

            def on_train_batch_end(
                self,
                trainer: ttl.nn.Trainer,
                batch: object,
                batch_idx: int,
                result: object,
            ) -> None:
                batch_ends.append((batch_idx, -1))

            def on_epoch_end(self, trainer: ttl.nn.Trainer, epoch: int) -> None:
                batch_ends.append((-2, epoch))

        config = ttl.nn.TrainerConfig(
            mode="benchmark",
            max_steps=2,
            max_epochs=1,
            compile_warmup_steps=0,
        )
        trainer = ttl.nn.Trainer(
            model=None,
            config=config,
            callbacks=[BatchTracker()],
        )
        trainer.fit(train_batches=[None, None])
        assert any(b[0] == -1 for b in batch_ends)
        assert any(b[0] >= 0 for b in batch_ends)
        assert any(b[0] == -2 for b in batch_ends)


class TestTrainerSchedulerEnvMode:
    """Trainer fit(env=...) runs reset/step loop and returns summary."""

    def test_scheduler_env_loop_summary(self) -> None:
        class MockEnv:
            def reset(
                self,
                seed: int | None = None,
                options: dict | None = None,
            ) -> tuple[dict, dict]:
                return {"step": 0}, {}

            def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
                return {"step": 1}, 1.0, True, False, {}

        config = ttl.nn.TrainerConfig(
            mode="scheduler_env",
            max_steps=3,
        )
        trainer = ttl.nn.Trainer(model=None, config=config)
        summary = trainer.fit(env=MockEnv())
        assert summary["steps"] >= 1
        assert "elapsed_s" in summary
        assert "metrics" in summary
        assert "total_reward" in summary["metrics"]
