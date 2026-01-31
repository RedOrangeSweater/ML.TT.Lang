# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lazy boot: boot code hits a placeholder -> exception -> load and patch -> retry.

This is a standalone example system (not part of tt-lang runtime). Use it to
load implementations on first use and optionally unload them later.
"""

from .core import (
    LazyLoadRequired,
    LazyWrapper,
    install_lazy_slot,
    make_placeholder,
)
from .loader import (
    load_callable_from_dotted_path,
    load_callable_from_module_path,
    make_dir_resolver,
    make_file_resolver,
    make_load_and_patch,
)

__all__ = [
    "LazyLoadRequired",
    "LazyWrapper",
    "install_lazy_slot",
    "make_placeholder",
    "load_callable_from_dotted_path",
    "load_callable_from_module_path",
    "make_dir_resolver",
    "make_file_resolver",
    "make_load_and_patch",
]
