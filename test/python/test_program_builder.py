# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Program layer: ProgramBuilder and run(program, *args, grid=...) contract.

Verifies ProgramBuilder builds ProgramSpec; run() requires grid when program is first arg.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

import pytest

from ttl.constants import MemorySpace
from ttl.program import (
    ProgramBuilder,
    ProgramDecoratorParams,
    ProgramOptions,
    ProgramSpec,
)


def test_program_builder_builds_spec():
    """ProgramBuilder.with_grid(...).with_options(...).build() returns ProgramSpec."""

    def _dummy_program(_x):
        pass

    spec = (
        ProgramBuilder(_dummy_program)
        .with_grid((2, 2))
        .with_options(memory_space=MemorySpace.L1)
        .build()
    )
    assert isinstance(spec, ProgramSpec)
    assert spec.program is _dummy_program
    assert spec.grid == (2, 2)
    assert spec.options.memory_space == MemorySpace.L1


def test_program_builder_default_options():
    """ProgramBuilder without with_options uses default ProgramOptions."""
    def _dummy_program(_x):
        pass

    spec = ProgramBuilder(_dummy_program).with_grid((1, 1)).build()
    assert spec.options.num_outs == 1
    assert spec.options.tiled is True


def test_run_requires_grid_when_program_first_arg():
    """run(program, *args) without grid= raises ValueError for undecorated programs."""
    from ttl import run

    def _dummy_program(_x):
        pass

    with pytest.raises(ValueError, match="grid= is required when passing program"):
        run(_dummy_program, None)


def test_run_program_uses_decorator_grid_when_missing():
    """run(program, *args) uses decorator params when grid is omitted."""
    from ttl import run

    def _dummy_program(_x):
        pass

    _dummy_program._ttl_program = True  # simulate @ttl.program-decorated
    _dummy_program._ttl_program_params = ProgramDecoratorParams(
        grid=(1, 1),
        options=ProgramOptions(),
    )
    with pytest.raises(ValueError, match="No threads found"):
        run(_dummy_program, None)
