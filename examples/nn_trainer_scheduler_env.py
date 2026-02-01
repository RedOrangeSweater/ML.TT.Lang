# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: ttl.nn.Trainer in scheduler_env mode (step/reward loop, no tt-device).

Uses a minimal Gymnasium-style env (reset/step); runs without hardware.
To use the real SchedulerPlacementEnv, set PYTHONPATH to include apps/scheduler-viz
and pass that env to Trainer.fit(env=...).
"""

import ttl.nn


class MockSchedulerEnv:
    """Minimal env: reset returns obs/info; step returns obs, reward, terminated, truncated, info."""

    def __init__(self, max_steps: int = 5):
        self._step_count = 0
        self._max_steps = max_steps

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        self._step_count = 0
        return {"step": 0}, {"info": "reset"}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1
        # Stub reward: small positive per step
        reward = 0.1 * (1 if 0 <= action < 10 else -1)
        terminated = self._step_count >= self._max_steps
        return (
            {"step": self._step_count},
            reward,
            terminated,
            False,
            {"action": action},
        )


if __name__ == "__main__":
    env = MockSchedulerEnv(max_steps=5)
    config = ttl.nn.TrainerConfig(
        mode="scheduler_env",
        max_steps=10,
        compile_warmup_steps=0,
    )
    trainer = ttl.nn.Trainer(model=None, config=config)
    summary = trainer.fit(env=env)
    print("Summary:", summary)
    print("Steps:", summary.get("steps"), "metrics:", summary.get("metrics"))
