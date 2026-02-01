# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DeviceManager (open/close, context manager) and TensorAdapter (torch<->ttnn, layout/memory defaults)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None  # type: ignore[assignment]


def _require_ttnn() -> Any:
    if ttnn is None:
        raise RuntimeError(
            "ttnn is required for DeviceManager and TensorAdapter. "
            "Install tt-metal / ttnn or run with TTLANG_COMPILE_ONLY=1."
        )
    return ttnn


def _is_torch_tensor(x: Any) -> bool:
    return torch is not None and hasattr(torch, "Tensor") and isinstance(x, torch.Tensor)


def _is_ttnn_tensor(x: Any) -> bool:
    if ttnn is None:
        return False
    return hasattr(ttnn, "Tensor") and isinstance(x, ttnn.Tensor)


class DeviceManager:
    """Context manager for ttnn device: open on enter, close on exit."""

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self._device: Any = None

    def open(self) -> Any:
        """Open device; return device handle. Raises if ttnn unavailable."""
        ttnn_mod = _require_ttnn()
        self._device = ttnn_mod.open_device(device_id=self.device_id)
        return self._device

    def close(self) -> None:
        """Close device if opened."""
        if self._device is not None and ttnn is not None:
            ttnn.close_device(self._device)
            self._device = None

    def __enter__(self) -> Any:
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def device(self) -> Any:
        """Current device handle; None if not opened."""
        return self._device


class TensorAdapter:
    """Convert torch tensors to/from ttnn with default layout and memory config.

    Defaults: TILE_LAYOUT, L1_MEMORY_CONFIG. Use when device is available;
    with TTLANG_COMPILE_ONLY=1, pass-through (no conversion) for compile-only path.
    """

    def __init__(
        self,
        device: Any = None,
        layout: Any = None,
        memory_config: Any = None,
        dtype: Any = None,
    ) -> None:
        """Build adapter with optional device and ttnn defaults.

        Args:
            device: ttnn device handle (optional; required for to_device).
            layout: ttnn layout (default TILE_LAYOUT when ttnn present).
            memory_config: ttnn memory config (default L1_MEMORY_CONFIG).
            dtype: ttnn dtype (default None = preserve torch dtype mapping).
        """
        self._device = device
        self._layout = layout
        self._memory_config = memory_config
        self._dtype = dtype

    def _ttnn_defaults(self) -> tuple[Any, Any, Any]:
        if ttnn is None:
            return None, None, None
        return (
            self._layout if self._layout is not None else ttnn.TILE_LAYOUT,
            self._memory_config if self._memory_config is not None else ttnn.L1_MEMORY_CONFIG,
            self._dtype,
        )

    def to_device(self, tensor: Any) -> Any:
        """Convert torch tensor to ttnn tensor on adapter device. Pass-through if already ttnn or no ttnn."""
        if ttnn is None or _is_ttnn_tensor(tensor):
            return tensor
        if not _is_torch_tensor(tensor):
            return tensor
        layout, mem_cfg, dtype = self._ttnn_defaults()
        kwargs: dict[str, Any] = {"layout": layout, "memory_config": mem_cfg}
        if self._device is not None:
            kwargs["device"] = self._device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ttnn.from_torch(tensor, **kwargs)

    def to_host(self, tensor: Any) -> Any:
        """Convert ttnn tensor to torch. Pass-through if already torch or no ttnn."""
        if ttnn is None or _is_torch_tensor(tensor):
            return tensor
        if _is_ttnn_tensor(tensor):
            return ttnn.to_torch(tensor)
        return tensor

    def adapt_args(self, *args: Any) -> tuple[Any, ...]:
        """Convert sequence of tensors to device (torch -> ttnn on device)."""
        return tuple(self.to_device(a) for a in args)

    def adapt_result(self, result: Any) -> Any:
        """Convert result to host (ttnn -> torch). Handles single tensor or tuple of tensors."""
        if isinstance(result, tuple):
            return tuple(self.to_host(x) for x in result)
        return self.to_host(result)
