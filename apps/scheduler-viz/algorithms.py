# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Algorithm registry for scheduler placement policies.

Register policies with register(id, fn); get(id) returns policy or raises.
Adding a new algorithm = one register() call, no if-chains in backend.
"""

from __future__ import annotations

from collections.abc import Callable


class UnknownAlgorithmError(ValueError):
    """Raised when algorithm_id is not in the registry."""

    def __init__(self, algorithm_id: str, known: list[str]) -> None:
        self.algorithm_id = algorithm_id
        self.known = known
        super().__init__(f"Unknown algorithm_id: {algorithm_id}. Use one of {known}")


PolicyFn = Callable[["SchedulerPlacementEnv"], int]


class AlgorithmRegistry:
    """Registry of placement policies: register(id, fn), get(id)."""

    def __init__(self) -> None:
        self._policies: dict[str, PolicyFn] = {}

    def register(self, algorithm_id: str, policy: PolicyFn) -> None:
        """Register a policy for the given id."""
        self._policies[algorithm_id] = policy

    def get(self, algorithm_id: str) -> PolicyFn:
        """Return policy for id; raises UnknownAlgorithmError if missing."""
        if algorithm_id not in self._policies:
            raise UnknownAlgorithmError(algorithm_id, list(self._policies.keys()))
        return self._policies[algorithm_id]

    def ids(self) -> list[str]:
        """Return list of registered algorithm ids."""
        return list(self._policies.keys())


# Global registry; algorithms register on import
ALGORITHMS = AlgorithmRegistry()


def register(algorithm_id: str) -> Callable[[PolicyFn], PolicyFn]:
    """Decorator to register a policy: @register(\"rcw\") def get_action_rcw(env): ..."""

    def decorator(fn: PolicyFn) -> PolicyFn:
        ALGORITHMS.register(algorithm_id, fn)
        return fn

    return decorator


# Built-in policies: use decorator so adding a new algorithm = one @register line
from scheduler_env import SchedulerPlacementEnv


@register("rcw")
def get_action_rcw(env: SchedulerPlacementEnv) -> int:
    """RCW preset: always place on core 0 (first core)."""
    return 0


@register("random")
def get_action_random(env: SchedulerPlacementEnv) -> int:
    """Random policy: sample from action space."""
    return int(env.action_space.sample())
