# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest: ttl.nn API (PipelineConfig, ProgramModule, Sequential).

Tests Pydantic validation, construction, and compile-only execution where applicable.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttl
from ttl.program import ProgramOptions


@pytest.fixture(autouse=True)
def compile_only() -> None:
    """Avoid device execution in tests."""
    os.environ["TTLANG_COMPILE_ONLY"] = "1"


class TestPipelineConfig:
    """PipelineConfig Pydantic validation and .options."""

    def test_default_options(self) -> None:
        cfg = ttl.nn.PipelineConfig()
        opts = cfg.options
        assert opts.memory_space == "L1"
        assert opts.tiled is True
        assert opts.num_outs == 1

    def test_custom_options(self) -> None:
        cfg = ttl.nn.PipelineConfig(
            options=ProgramOptions(memory_space="L1", objective="latency")
        )
        opts = cfg.options
        assert opts.memory_space == "L1"
        assert opts.objective == "latency"


class TestProgramModule:
    """ProgramModule construction and call (compile-only)."""

    @ttl.program(grid=(1, 1))
    @staticmethod
    def _add_kernel(lhs, rhs, out):
        lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
        rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
        out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def add_compute():
            l = lhs_cb.wait()
            r = rhs_cb.wait()
            o = out_cb.reserve()
            o.store(l + r)
            lhs_cb.pop()
            rhs_cb.pop()
            out_cb.push()

        @ttl.datamovement()
        def dm_read():
            lhs_blk = lhs_cb.reserve()
            ttl.copy(lhs[0, 0], lhs_blk).wait()
            lhs_cb.push()
            rhs_blk = rhs_cb.reserve()
            ttl.copy(rhs[0, 0], rhs_blk).wait()
            rhs_cb.push()

        @ttl.datamovement()
        def dm_write():
            out_blk = out_cb.wait()
            ttl.copy(out_blk, out[0, 0]).wait()
            out_cb.pop()

    def test_construction(self) -> None:
        mod = ttl.nn.ProgramModule(self._add_kernel, grid=(1, 1))
        assert mod is not None

    def test_call_compile_only(self) -> None:
        mod = ttl.nn.ProgramModule(self._add_kernel, grid=(1, 1))
        lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out = torch.zeros((32, 32), dtype=torch.bfloat16)
        result = mod(lhs, rhs, out)
        assert result is None  # compile-only, no device run


class TestSequential:
    """Sequential / Pipeline construction and single-step call."""

    @ttl.program(grid=(1, 1))
    @staticmethod
    def _add_kernel(lhs, rhs, out):
        lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
        rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
        out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def add_compute():
            l = lhs_cb.wait()
            r = rhs_cb.wait()
            o = out_cb.reserve()
            o.store(l + r)
            lhs_cb.pop()
            rhs_cb.pop()
            out_cb.push()

        @ttl.datamovement()
        def dm_read():
            lhs_blk = lhs_cb.reserve()
            ttl.copy(lhs[0, 0], lhs_blk).wait()
            lhs_cb.push()
            rhs_blk = rhs_cb.reserve()
            ttl.copy(rhs[0, 0], rhs_blk).wait()
            rhs_cb.push()

        @ttl.datamovement()
        def dm_write():
            out_blk = out_cb.wait()
            ttl.copy(out_blk, out[0, 0]).wait()
            out_cb.pop()

    def test_sequential_single_module(self) -> None:
        seq = ttl.nn.Sequential(
            ttl.nn.ProgramModule(self._add_kernel, grid=(1, 1)),
        )
        lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out = torch.zeros((32, 32), dtype=torch.bfloat16)
        result = seq(lhs, rhs, out)
        assert result is None  # compile-only

    def test_pipeline_alias(self) -> None:
        pipe = ttl.nn.Pipeline(
            ttl.nn.ProgramModule(self._add_kernel, grid=(1, 1)),
        )
        assert isinstance(pipe, ttl.nn.Sequential)
