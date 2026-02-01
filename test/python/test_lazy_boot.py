# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the lazy boot system (examples/lazy_boot).

Boot code hits placeholder -> LazyLoadRequired -> load_and_patch -> retry from start of function.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add examples to path so we can import lazy_boot (standalone example under examples/)
_examples = Path(__file__).resolve().parents[2] / "examples"
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

from lazy_boot import (  # type: ignore[import-not-found]
    LazyLoadRequired,
    LazyWrapper,
    make_load_and_patch,
    make_placeholder,
)
from lazy_boot.loader import (
    load_callable_from_module_path,  # type: ignore[import-not-found]
)


def test_lazy_load_required_attributes() -> None:
    mod = SimpleNamespace(__name__="test_mod")
    e = LazyLoadRequired(mod, "foo", (1, 2), {"k": 3})
    assert e.module is mod
    assert e.symbol == "foo"
    assert e.args == (1, 2)
    assert e.kwargs == {"k": 3}


def test_make_placeholder_raises() -> None:
    mod = SimpleNamespace(__name__="test_mod")
    placeholder = make_placeholder(mod, "foo")
    with pytest.raises(LazyLoadRequired) as exc_info:
        placeholder(1, 2, k=3)
    assert exc_info.value.module is mod
    assert exc_info.value.symbol == "foo"
    assert exc_info.value.args == (1, 2)
    assert exc_info.value.kwargs == {"k": 3}


def test_load_and_patch_retry(tmp_path: Path) -> None:
    """Write a small module, resolve to it, load_and_patch installs impl and retry runs it."""
    impl_file = tmp_path / "my_impl.py"
    impl_file.write_text("def my_impl(a: int, b: int) -> int:\n    return a * b\n")

    mod = SimpleNamespace(__name__="test_mod")
    loaded: list[str] = []

    def resolve(_m: object, symbol: str):  # type: ignore[no-untyped-def]
        if symbol == "my_impl":
            loaded.append(symbol)
            return load_callable_from_module_path(impl_file, "my_impl")
        return None

    load_and_patch = make_load_and_patch(resolve)
    placeholder = make_placeholder(mod, "my_impl")
    wrapper = LazyWrapper(mod, "my_impl", placeholder, load_and_patch)
    mod.my_impl = wrapper

    result = mod.my_impl(3, 4)
    assert result == 12
    assert loaded == ["my_impl"]

    # Second call uses cached impl (resolve not called again)
    result2 = mod.my_impl(2, 5)
    assert result2 == 10
    assert loaded == ["my_impl"]


def test_boot_demo_run() -> None:
    """Run the boot demo: lazy slots load from impls/ on first call."""
    from lazy_boot.boot_demo import run  # type: ignore[import-not-found]

    run()
