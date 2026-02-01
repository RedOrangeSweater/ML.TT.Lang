# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime tensor abstraction: proxy layer for ttnn.Tensor.

Types are defined here; the rest of the code uses these types, not ttnn.Tensor.
When ttnn is installed, ttnn.Tensor is registered as a runtime tensor type and
validated at module load. When ttnn is not installed, no type is registered and
is_ttnn_tensor() returns False. Validation runs only at initialization.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .boundary import TtnnDeviceLike

# Types that implement "runtime tensor" (device tensor we can run kernels on).
# Populated at module load when ttnn is available; validated then if ttnn present.
_RUNTIME_TENSOR_TYPES: set[type] = set()


@runtime_checkable
class RuntimeTensor(Protocol):
    """Protocol for tensors that can be used at runtime (device tensors).

    Implementations (e.g. ttnn.Tensor when ttnn is installed) must provide
    device() and memory_config(). No ttnn import here; validation at init.
    """

    def device(self) -> TtnnDeviceLike:
        """Return device handle (e.g. for grid/bounding_box)."""
        ...

    def memory_config(self) -> object:
        """Return memory config (buffer_type, memory_layout for L1/DRAM, interleaved)."""
        ...


def _validate_runtime_tensor_type(t: type) -> None:
    """Check that type t has device() and memory_config(). No ttnn import."""
    if not hasattr(t, "device") or not callable(getattr(t, "device")):
        raise TypeError(f"{t!r} has no callable device()")
    if not hasattr(t, "memory_config") or not callable(getattr(t, "memory_config")):
        raise TypeError(f"{t!r} has no callable memory_config()")


def is_ttnn_tensor(obj: object) -> bool:
    """Return True if obj is a registered TTNN tensor (e.g. ttnn.Tensor when ttnn present).

    No ttnn import here; uses _RUNTIME_TENSOR_TYPES populated at module load.
    When ttnn is not installed, set is empty and this always returns False.
    """
    return any(isinstance(obj, t) for t in _RUNTIME_TENSOR_TYPES)


def _init_ttnn_if_available() -> None:
    """Register ttnn.Tensor as runtime tensor when ttnn is available; validate at init."""
    try:
        import ttnn  # type: ignore[import-untyped]

        _validate_runtime_tensor_type(ttnn.Tensor)
        _RUNTIME_TENSOR_TYPES.add(ttnn.Tensor)
    except (ModuleNotFoundError, ImportError):
        pass


# Run at module load: register ttnn.Tensor when ttnn is installed, validate then.
_init_ttnn_if_available()
