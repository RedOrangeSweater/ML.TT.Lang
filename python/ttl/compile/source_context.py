# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Source context for kernel compilation: file, lines, and capture collection."""

from __future__ import annotations

import inspect
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from pykernel._src.utils import _cleanup_source_code

from ..circular_buffer import CircularBuffer
from ..dtype_utils import is_ttnn_tensor
from .pipeline import _get_source_line_offset


class CompilationSourceContext(BaseModel):
    """Pydantic context for a single kernel compilation: source file, lines, and debug flags.

    Built once from the decorated function so call sites do not manually pass
    _source_file, _source_lines, _line_offset, debug_locations through kwargs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_file: str = "<unknown>"
    source_lines: list[str] = Field(default_factory=list)
    line_offset: int = 0
    debug_locations: bool = True
    verbose: bool = False
    source_code: str = ""
    fn_globals: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_function(
        cls, f: Callable[..., object], *, verbose: bool = False
    ) -> CompilationSourceContext:
        """Build context from a decorated function (captures file, source, line offset, globals)."""
        try:
            source_file = inspect.getfile(f)
        except (TypeError, OSError):
            source_file = "<unknown>"
        source_code = _cleanup_source_code(f)
        source_lines = source_code.splitlines()
        line_offset = _get_source_line_offset(f)
        fn_globals = getattr(f, "__globals__", {})
        return cls(
            source_file=source_file,
            source_lines=source_lines,
            line_offset=line_offset,
            debug_locations=True,
            verbose=verbose,
            source_code=source_code,
            fn_globals=fn_globals,
        )


def collect_captures(
    f: Callable[..., object],
) -> dict[str, int | CircularBuffer]:
    """
    Collect and convert captured variables from function closure.

    Returns:
        Dictionary mapping variable names to converted values (int, ttnn tensor, CircularBuffer).
    """
    if f.__closure__ is None:
        return {}

    def convert(name: str, val: object) -> int | object:
        if isinstance(val, int):
            return val
        if is_ttnn_tensor(val):
            return val
        if isinstance(val, CircularBuffer):
            return val
        try:
            import torch
            if isinstance(val, torch.Tensor):
                return val
        except ImportError:
            pass
        raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }
