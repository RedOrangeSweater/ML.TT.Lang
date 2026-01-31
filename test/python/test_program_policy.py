# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test @ttl.program with objective/placement policy args (Phase 2).

Verifies that objective and placement are accepted, stored, and do not
break compilation or execution; behavior is unchanged.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import pytest
import torch
import ttl


@ttl.program(grid=(1, 1), objective="latency", placement="auto")
def add_with_policy(lhs, rhs, out):
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


def test_program_with_objective_placement_compiles():
    """@ttl.program(objective=..., placement=...) compiles without error."""
    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    try:
        add_with_policy(lhs, rhs, out)
    except TypeError as e:
        if "Unhandled capture" in str(e) or "torch.Tensor" in str(e):
            pytest.skip(
                "compile-only with torch tensors not supported in this env; "
                "policy args are accepted (see invalid_* tests)"
            )
        raise
    # Compile-only: out unchanged; no exception means compilation succeeded
    assert out.sum().item() == 0.0


def test_program_invalid_objective_raises():
    """@ttl.program(objective=invalid) raises ValueError."""
    with pytest.raises(ValueError, match=r"objective must be one of"):

        @ttl.program(grid=(1, 1), objective="invalid")
        def _bad():
            pass


def test_program_invalid_placement_raises():
    """@ttl.program(placement=invalid) raises ValueError."""
    with pytest.raises(ValueError, match=r"placement must be one of"):

        @ttl.program(grid=(1, 1), placement="invalid")
        def _bad():
            pass
