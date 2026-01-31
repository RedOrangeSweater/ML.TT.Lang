# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test E2E scheduler path (Phase 3-4): op graph + topology + stub.

With TTLANG_USE_SCHEDULER=1, compilation builds graph and runs scheduler stub;
result of compilation/execution must be unchanged.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"
os.environ["TTLANG_USE_SCHEDULER"] = "1"

import pytest
import torch
import ttl


@ttl.program(grid=(1, 1))
def add_with_scheduler(lhs, rhs, out):
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


def test_scheduler_path_compiles():
    """With TTLANG_USE_SCHEDULER=1, simple add compiles without error."""
    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    try:
        add_with_scheduler(lhs, rhs, out)
    except TypeError as e:
        if "Unhandled capture" in str(e) or "torch.Tensor" in str(e):
            pytest.skip("compile-only with torch tensors not supported in this env")
        raise
    assert out.sum().item() == 0.0


def test_op_graph_and_stub_unit():
    """Unit test: build op graph from thread_infos, run stub, check plan."""
    from ttl.scheduler import (
        build_op_graph_from_threads,
        build_topology_from_grid,
        schedule_stub,
    )

    thread_infos = [
        ("add_compute", "compute"),
        ("dm_read", "datamovement"),
        ("dm_write", "datamovement"),
    ]
    graph = build_op_graph_from_threads(thread_infos)
    assert len(graph.nodes) == 3
    assert graph.nodes["add_compute"].op_type == "elementwise"
    assert graph.nodes["dm_read"].op_type == "load"
    assert graph.nodes["dm_write"].op_type == "store"
    assert "add_compute" in graph.nodes["dm_read"].successors
    assert "dm_write" in graph.nodes["add_compute"].successors

    topology = build_topology_from_grid((1, 1))
    assert topology.num_cores() == 1
    plan = schedule_stub(graph, topology)
    assert plan.grid_cols == 1 and plan.grid_rows == 1
    for nid in graph.node_ids_in_order():
        assert plan.core_for(nid) == (0, 0)


def test_export_scheduler_input():
    """Export scheduler input returns JSON-serializable dict (doc 16)."""
    import json

    from ttl.scheduler import export_scheduler_input, export_scheduler_input_to_json

    thread_infos = [
        ("add_compute", "compute"),
        ("dm_read", "datamovement"),
        ("dm_write", "datamovement"),
    ]
    data = export_scheduler_input(thread_infos, (1, 1))
    assert "op_graph" in data and "topology" in data and "plan" in data
    assert len(data["op_graph"]["nodes"]) == 3
    assert data["topology"]["grid_cols"] == 1 and data["topology"]["grid_rows"] == 1
    assert len(data["plan"]["placement"]) == 3
    # Must be JSON-serializable
    json_str = json.dumps(data)
    loaded = json.loads(json_str)
    assert loaded["plan"]["placement"]["add_compute"] == [0, 0]

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        export_scheduler_input_to_json(thread_infos, (1, 1), path)
        with open(path) as fp:
            file_data = json.load(fp)
        assert file_data["op_graph"]["nodes"][0]["op_type"] in (
            "load",
            "store",
            "elementwise",
        )
    finally:
        os.unlink(path)


def test_program_style_scheduler_input():
    """Program-style (toy_program_add) yields scheduler input with same structure as stub (doc 16)."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.toy_program_add import simple_add

    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    simple_add(lhs, rhs, out)
    kernel = getattr(simple_add, "_last_compiled_kernel", None)
    if kernel is None:
        pytest.skip("_last_compiled_kernel not set (compile path)")
    data = kernel.get_scheduler_input()
    assert "op_graph" in data and "topology" in data and "plan" in data
    assert len(data["op_graph"]["nodes"]) == 3
    assert data["topology"]["grid_cols"] == 1 and data["topology"]["grid_rows"] == 1
    # Thread names from program: add_compute, dm_read, dm_write
    node_ids = [n["id"] for n in data["op_graph"]["nodes"]]
    assert "add_compute" in node_ids
    assert "dm_read" in node_ids
    assert "dm_write" in node_ids
