# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Source collector: extract source info and CB configs from compiled threads.

Builds all_source_files, all_source_lines, kernel_line_offsets from TTLGenericCompiler
threads; extracts CircularBuffer configs from thread closures.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import CellType

from pydantic import BaseModel, ConfigDict

from .._src.tensor_registry import register_tensor_source
from .._src.ttl_ast import TTLGenericCompiler
from ..circular_buffer import CircularBuffer
from ..diagnostics import find_variable_assignment
from ..dtype_utils import is_ttnn_tensor


class ThreadWrapperView(BaseModel):
    """Typed view of a decorated thread: wrapped callable and its closure."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    wrapped: Callable[..., object] | None = None
    closure: tuple[CellType, ...] | None = None

    @classmethod
    def from_callable(cls, thread_fn: Callable[..., object]) -> ThreadWrapperView:
        wrapped = getattr(thread_fn, "__wrapped__", None)
        closure = getattr(wrapped, "__closure__", None) if wrapped else None
        if closure is not None:
            closure = tuple(closure)
        return cls(wrapped=wrapped, closure=closure)


def get_source_line_offset(f: Callable[..., object]) -> int:
    """Get the line offset to convert parsed AST line numbers to actual file lines."""
    try:
        raw_lines, start_lineno = inspect.getsourcelines(f)
        num_decorator_lines = 0
        for line in raw_lines:
            stripped = line.strip()
            if stripped.startswith("@"):
                num_decorator_lines += 1
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                break
        return start_lineno + num_decorator_lines - 1
    except (TypeError, OSError):
        return 0


def collect_cb_configs(
    threads: list[Callable[..., object]],
) -> list[CircularBuffer | None]:
    """Extract CircularBuffer objects from thread closures, indexed by cb_index."""
    cb_configs_dict: dict[int, CircularBuffer] = {}
    for thread_fn in threads:
        view = ThreadWrapperView.from_callable(thread_fn)
        if not view.closure:
            continue
        for cell in view.closure:
            val = cell.cell_contents
            if isinstance(val, CircularBuffer):
                cb_configs_dict[val._cb_index] = val
    if not cb_configs_dict:
        return []
    max_idx = max(cb_configs_dict.keys())
    return [cb_configs_dict.get(i) for i in range(max_idx + 1)]


def collect_source_info_from_threads(
    threads: list[TTLGenericCompiler],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    """Build all_source_files, all_source_lines, kernel_line_offsets from compiled threads."""
    all_source_files: dict[str, str] = {}
    all_source_lines: dict[str, list[str]] = {}
    kernel_line_offsets: dict[str, int] = {}
    for ct in threads:
        info = ct.source_info
        all_source_files[ct.name] = info.source_file
        all_source_lines[ct.name] = info.source_lines
        kernel_line_offsets[ct.name] = info.line_offset
    return all_source_files, all_source_lines, kernel_line_offsets


def track_tensor_sources(
    f_params: object,
    args: tuple[object, ...],
    source_file: str,
) -> None:
    """Track source locations for tensor arguments."""
    if source_file == "<unknown>":
        return
    try:
        with open(source_file) as sf:
            source_lines = sf.read().splitlines()
    except OSError:
        return
    call_line = None
    for frame_info in inspect.stack():
        if frame_info.filename == source_file:
            call_line = frame_info.lineno
            break
    if call_line is None:
        return
    param_names = (
        list(f_params)
        if hasattr(f_params, "__iter__") and not isinstance(f_params, (str, bytes))
        else []
    )
    for param_name, arg in zip(param_names, args, strict=False):
        if not is_ttnn_tensor(arg):
            continue
        assign_line = find_variable_assignment(source_lines, param_name, call_line)
        if assign_line:
            register_tensor_source(arg, source_file, assign_line)


__all__ = [
    "ThreadWrapperView",
    "get_source_line_offset",
    "collect_cb_configs",
    "collect_source_info_from_threads",
    "track_tensor_sources",
]
