# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Primitive registry: register and lookup primitives by name (PyTorch nn.Module-style).

Enables extensibility without editing core facade.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .base import Primitive

T = TypeVar("T", bound=type)


class PrimitiveRegistry:
    """Registry for primitives (like torch.nn module registry)."""

    _primitives: dict[str, type[Primitive]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """Decorator to register a primitive class by name."""

        def _decorator(prim_class: T) -> T:
            cls._primitives[name] = prim_class  # type: ignore[assignment]
            return prim_class

        return _decorator

    @classmethod
    def get(cls, name: str) -> type[Primitive] | None:
        """Return registered primitive class by name, or None."""
        return cls._primitives.get(name)

    @classmethod
    def list_names(cls) -> list[str]:
        """Return all registered primitive names."""
        return list(cls._primitives.keys())


__all__ = ["PrimitiveRegistry"]
