# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttl.nn.Trainer: fit/benchmark and scheduler_env loops with callbacks and logging."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TypeVar

from .callbacks import Callback
from .config import TrainerConfig
from .logger import Logger, StdoutLogger

T = TypeVar("T", covariant=True)


@runtime_checkable
class ModelLike(Protocol[T]):
    """Protocol for models that can be trained or run."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        ...


@runtime_checkable
class SchedulerEnvLike(Protocol):
    """Protocol for scheduler env: reset and step (Gymnasium-style)."""

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        ...


class Trainer:
    """Trainer for benchmark/inference loop and scheduler_env (reward-based) loop.

    Phases: setup -> compile_warmup -> run loop -> teardown. Calls callbacks and logger.
    """

    def __init__(
        self,
        model: ModelLike | None = None,
        config: TrainerConfig | None = None,
        callbacks: list[Callback] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.model = model
        self.config = TrainerConfig() if config is None else config
        self.callbacks = [] if callbacks is None else list(callbacks)
        self.logger = StdoutLogger() if logger is None else logger
        self._global_step = 0

    def fit(
        self,
        train_batches: list[object] | object | None = None,
        val_batches: list[object] | object | None = None,
        env: SchedulerEnvLike | None = None,
    ) -> dict[str, object]:
        """Run fit: compile warmup then benchmark loop or scheduler_env loop.

        train_batches: list of batches for benchmark (or single batch, repeated).
        env: when config.mode == "scheduler_env", run reset/step loop on this env.
        Returns summary dict (steps, elapsed_s, metrics).
        """
        summary: dict[str, object] = {"steps": 0, "elapsed_s": 0.0, "metrics": {}}
        self._global_step = 0
        prev_auto_profile: str | None = os.environ.get("TTLANG_AUTO_PROFILE")
        try:
            if self.config.enable_profiling:
                os.environ["TTLANG_AUTO_PROFILE"] = "1"
            self._invoke_callbacks("on_fit_start")

            # Compile warmup
            self._invoke_callbacks("on_compile_start")
            self._compile_warmup(train_batches)
            self._invoke_callbacks("on_compile_end")

            # Run loop
            if self.config.mode == "scheduler_env" and env is not None:
                summary = self._run_scheduler_env_loop(env, summary)
            else:
                summary = self._run_benchmark_loop(train_batches, summary)

            self._invoke_callbacks("on_fit_end")
            if hasattr(self.logger, "flush"):
                self.logger.flush()
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = out_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            return summary
        finally:
            if self.config.enable_profiling:
                if prev_auto_profile is not None:
                    os.environ["TTLANG_AUTO_PROFILE"] = prev_auto_profile
                else:
                    os.environ.pop("TTLANG_AUTO_PROFILE", None)

    def _invoke_callbacks(self, method: str, *args: object, **kwargs: object) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if fn is not None and callable(fn):
                fn(self, *args, **kwargs)

    def _compile_warmup(self, train_batches: object) -> None:
        n = self.config.compile_warmup_steps
        if not n or self.model is None:
            return
        batches = _as_batch_list(train_batches, n)
        for i in range(n):
            batch = batches[i] if i < len(batches) else (batches[0] if batches else None)
            self._run_one_step(batch, i, warmup=True)

    def _run_benchmark_loop(
        self,
        train_batches: object,
        summary: dict[str, object],
    ) -> dict[str, object]:
        max_steps = self.config.max_steps
        max_epochs = self.config.max_epochs
        if max_steps is not None and max_steps <= 0:
            return summary
        batches = _as_batch_list(train_batches, max_steps or max_epochs or 1)
        if not batches and self.model is not None:
            batches = [None]
        steps = 0
        latencies: list[float] = []
        start = time.perf_counter()
        for epoch in range(max_epochs or 1):
            self._invoke_callbacks("on_epoch_start", epoch)
            for i, batch in enumerate(batches):
                if max_steps is not None and steps >= max_steps:
                    break
                self._invoke_callbacks("on_train_batch_start", batch, steps)
                t0 = time.perf_counter()
                result = self._run_one_step(batch, steps, warmup=False)
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)
                self._invoke_callbacks("on_train_batch_end", batch, steps, result)
                if self.logger and latencies:
                    self.logger.log_metrics(
                        {"latency_s": elapsed, "step": steps},
                        step=steps,
                        prefix="benchmark",
                    )
                steps += 1
            self._invoke_callbacks("on_epoch_end", epoch)
            if max_steps is not None and steps >= max_steps:
                break
        total = time.perf_counter() - start
        summary["steps"] = steps
        summary["elapsed_s"] = total
        if latencies:
            summary["metrics"] = {
                "latency_mean_s": sum(latencies) / len(latencies),
                "latency_min_s": min(latencies),
                "latency_max_s": max(latencies),
                "throughput_steps_per_s": steps / total if total > 0 else 0,
            }
        return summary

    def _run_scheduler_env_loop(
        self,
        env: SchedulerEnvLike,
        summary: dict[str, object],
    ) -> dict[str, object]:
        obs, info = env.reset()
        steps = 0
        total_reward = 0.0
        start = time.perf_counter()
        max_steps = self.config.max_steps or 1000
        while steps < max_steps:
            action = 0  # Default: place on core 0 (user can replace with policy)
            if hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if self.logger:
                self.logger.log_metrics(
                    {"reward": reward, "total_reward": total_reward},
                    step=steps,
                    prefix="scheduler_env",
                )
            if terminated or truncated:
                obs, info = env.reset()
        total = time.perf_counter() - start
        summary["steps"] = steps
        summary["elapsed_s"] = total
        summary["metrics"] = {
            "total_reward": total_reward,
            "steps_per_s": steps / total if total > 0 else 0,
        }
        return summary

    def _run_one_step(
        self,
        batch: object,
        batch_idx: int,
        warmup: bool = False,
    ) -> object:
        if self.model is None:
            return None
        if hasattr(self.model, "training_step"):
            return self.model.training_step(batch, batch_idx)
        if batch is None:
            return self.model()
        if isinstance(batch, (list, tuple)):
            return self.model(*batch)
        return self.model(batch)


def _as_batch_list(batches: list[object] | object | None, max_len: int) -> list[object]:
    """Normalize train_batches to a list of length up to max_len."""
    if batches is None:
        return []
    if isinstance(batches, list):
        return batches[:max_len] if max_len else batches
    return [batches] * min(1, max_len) if max_len else [batches]
