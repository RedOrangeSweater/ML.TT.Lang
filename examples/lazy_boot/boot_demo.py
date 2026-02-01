# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boot demo: minimal boot module with lazy slots; first call triggers load and retry.

Run from repo root:
  python -c "import sys; sys.path.insert(0, 'examples'); from lazy_boot.boot_demo import run; run()"
  or: python examples/lazy_boot/boot_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure examples/lazy_boot is importable
_here = Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from lazy_boot import install_lazy_slot, make_dir_resolver, make_load_and_patch


def run() -> None:
    """Boot: install lazy slots, then call foo and bar; first call loads from impls/."""
    # Boot module: the namespace we attach lazy slots to (this module)
    boot_module = sys.modules[__name__]

    # Resolver: load callable from impls/{symbol}.py
    impls_dir = _here / "impls"
    resolve = make_dir_resolver(impls_dir)
    load_and_patch = make_load_and_patch(resolve)

    # Install lazy slots so that first call raises LazyLoadRequired -> load -> retry
    install_lazy_slot(boot_module, "foo", load_and_patch)
    install_lazy_slot(boot_module, "bar", load_and_patch)

    # Get the lazy slots (wrappers) and call them; first call loads from impls/
    foo = boot_module.foo
    bar = boot_module.bar

    result_foo = foo(2, 3)  # placeholder -> load foo from impls/foo.py -> retry -> 5
    result_bar = bar(
        "world"
    )  # placeholder -> load bar from impls/bar.py -> retry -> "Hello, world!"

    assert result_foo == 5, result_foo
    assert result_bar == "Hello, world!", result_bar
    print(f"foo(2, 3) = {result_foo}")
    print(f"bar('world') = {result_bar}")

    # Second call uses cached implementation (no load)
    assert foo(1, 1) == 2
    assert bar("boot") == "Hello, boot!"
    print("Second calls used cached impl.")


if __name__ == "__main__":
    run()
