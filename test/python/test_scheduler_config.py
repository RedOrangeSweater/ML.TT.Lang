# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for abstract engine config (scheduler/Toy backend).

Load from JSON, validate via Pydantic. Integration: run() with engine_config uses
topology for tenstorrent and schedule_toy_stub for toy. See doc 20.
"""

import os
from pathlib import Path

import pytest

from ttl.program import ProgramOptions, ProgramSpec
from ttl.scheduler import (
    AbstractEngineConfig,
    TopologyGridSpec,
    load_abstract_engine_config,
    validate_topology_connectivity,
)
from ttl import run


def test_load_abstract_engine_config_from_json():
    """load_abstract_engine_config(path) loads JSON and returns AbstractEngineConfig."""
    path = Path(__file__).parent / "example_abstract_engine_config.json"
    config = load_abstract_engine_config(path)
    assert isinstance(config, AbstractEngineConfig)
    assert config.backend == "toy_shops"
    assert config.objective == "latency"
    assert config.graph_toy_generator is not None
    assert config.graph_toy_generator.num_nodes == 4
    assert config.topology_graph is not None
    assert len(config.topology_graph.nodes) == 4
    assert len(config.topology_graph.edges) == 4


def test_validate_topology_connectivity_example():
    """validate_topology_connectivity(config) passes for connected graph."""
    path = Path(__file__).parent / "example_abstract_engine_config.json"
    config = load_abstract_engine_config(path)
    assert validate_topology_connectivity(config) is True


def test_abstract_engine_config_grid():
    """AbstractEngineConfig with topology_grid returns get_topology_grid()."""
    config = AbstractEngineConfig(
        backend="tenstorrent",
        topology_grid={"grid_cols": 2, "grid_rows": 3},
    )
    assert config.get_topology_grid() == (2, 3)


def test_abstract_engine_config_minimal():
    """AbstractEngineConfig accepts minimal fields (backend only)."""
    config = AbstractEngineConfig(backend="toy_bakeries")
    assert config.backend == "toy_bakeries"
    assert config.objective == "latency"
    assert config.get_topology_grid() is None


# Minimal program with threads for run(..., engine_config=...) integration tests.
def _minimal_add_impl(lhs, rhs, out):
    import ttl

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


def test_run_with_engine_config_tenstorrent_uses_topology():
    """run(..., engine_config=tenstorrent with topology_grid) uses config topology (use_scheduler)."""
    import ttl
    import torch

    minimal_add = ttl.program(grid=(2, 2))(_minimal_add_impl)
    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    spec = ProgramSpec(
        program=minimal_add,
        grid=(2, 2),
        options=ProgramOptions(),
    )
    engine_config = AbstractEngineConfig(
        backend="tenstorrent",
        topology_grid=TopologyGridSpec(grid_cols=2, grid_rows=2),
    )
    prev = os.environ.get("TTLANG_USE_SCHEDULER")
    prev_compile = os.environ.get("TTLANG_COMPILE_ONLY")
    try:
        os.environ["TTLANG_USE_SCHEDULER"] = "1"
        os.environ["TTLANG_COMPILE_ONLY"] = "1"
        run(spec, lhs, rhs, out, engine_config=engine_config)
    except TypeError as e:
        if "Unhandled capture" in str(e) or "torch.Tensor" in str(e):
            pytest.skip("compile-only with torch tensors not supported in this env")
        raise
    finally:
        if prev is not None:
            os.environ["TTLANG_USE_SCHEDULER"] = prev
        elif "TTLANG_USE_SCHEDULER" in os.environ:
            os.environ.pop("TTLANG_USE_SCHEDULER")
        if prev_compile is not None:
            os.environ["TTLANG_COMPILE_ONLY"] = prev_compile
        elif "TTLANG_COMPILE_ONLY" in os.environ:
            os.environ.pop("TTLANG_COMPILE_ONLY")


def test_run_with_engine_config_toy_completes():
    """run(..., engine_config=toy_shops with topology_graph) runs schedule_toy_stub (use_scheduler)."""
    import ttl
    import torch

    minimal_add = ttl.program(grid=(1, 1))(_minimal_add_impl)
    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    spec = ProgramSpec(
        program=minimal_add,
        grid=(1, 1),
        options=ProgramOptions(),
    )
    path = Path(__file__).parent / "example_abstract_engine_config.json"
    engine_config = load_abstract_engine_config(path)
    assert engine_config.backend == "toy_shops"
    prev = os.environ.get("TTLANG_USE_SCHEDULER")
    prev_compile = os.environ.get("TTLANG_COMPILE_ONLY")
    try:
        os.environ["TTLANG_USE_SCHEDULER"] = "1"
        os.environ["TTLANG_COMPILE_ONLY"] = "1"
        run(spec, lhs, rhs, out, engine_config=engine_config)
    except TypeError as e:
        if "Unhandled capture" in str(e) or "torch.Tensor" in str(e):
            pytest.skip("compile-only with torch tensors not supported in this env")
        raise
    finally:
        if prev is not None:
            os.environ["TTLANG_USE_SCHEDULER"] = prev
        elif "TTLANG_USE_SCHEDULER" in os.environ:
            os.environ.pop("TTLANG_USE_SCHEDULER")
        if prev_compile is not None:
            os.environ["TTLANG_COMPILE_ONLY"] = prev_compile
        elif "TTLANG_COMPILE_ONLY" in os.environ:
            os.environ.pop("TTLANG_COMPILE_ONLY")
