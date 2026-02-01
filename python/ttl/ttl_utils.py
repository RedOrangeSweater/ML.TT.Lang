# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for tt-lang."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["get_thread_type_string", "tmp_dir"]


@contextmanager
def tmp_dir(
    base_path: Path, dir_name: str | Callable[[], str]
) -> Generator[Path, None, None]:
    """
    Create a subdirectory under base_path and yield its path.

    dir_name may be a string or a callable (e.g. lambda with uuid) evaluated on
    entering the context. The directory is not removed on exit so yielded paths
    remain valid. Caller must ensure base_path exists (e.g. base_path.mkdir(
    parents=True, exist_ok=True)).
    """
    name = dir_name() if callable(dir_name) else dir_name
    dir_path = base_path / name
    dir_path.mkdir(parents=True, exist_ok=False)
    try:
        yield dir_path
    finally:
        pass  # Keep dir so paths handed out remain valid


# Mapping from kernel type strings to thread type strings
_KERNEL_TYPE_TO_THREAD_TYPE = {
    "compute": "compute",
    "datamovement": "noc",
    "ethernet": "ethernet",
}


def get_thread_type_string(input: str | object) -> str:
    """Map kernel type to thread type string.

    Handles both string kernel types and MLIR ThreadTypeAttr.

    Args:
        input: Either a string kernel type ("compute", "datamovement", "ethernet")
               or a ttkernel.ThreadTypeAttr from MLIR IR

    Returns:
        Thread type string: "compute", "noc", "ethernet"

    Raises:
        ValueError: If input is a string that's not a valid kernel type
    """
    # If it's already a string, use the dict lookup
    if isinstance(input, str):
        if input in _KERNEL_TYPE_TO_THREAD_TYPE:
            return _KERNEL_TYPE_TO_THREAD_TYPE[input]
        raise ValueError(f"Unknown kernel type: {input}")

    # For ThreadTypeAttr objects, parse the string representation
    # ThreadTypeAttr prints as #ttkernel.thread<compute> or #ttkernel.thread<noc>
    input_str = str(input)
    for thread_type in ["compute", "noc", "ethernet"]:
        if thread_type in input_str:
            return thread_type

    raise ValueError(f"Unknown thread type in attribute: {input_str}")
