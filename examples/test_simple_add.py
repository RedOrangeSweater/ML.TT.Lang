# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple add test that exercises the full copy/wait path under simulation.

This example:
- copies A and B tiles into L1 circular buffers (async copy + wait)
- computes C = A + B
- copies the result tile back to output (async copy + wait)
"""

import ttl
import ttnn
from sim.testing import assert_pcc


@ttl.kernel(grid=(1, 1))
def simple_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    assert a_in.shape == b_in.shape == out.shape

    # One 32x32 tile (single-tile CBs).
    a_cb = ttl.make_circular_buffer_like(a_in, shape=(1, 1), buffer_factor=1)
    b_cb = ttl.make_circular_buffer_like(b_in, shape=(1, 1), buffer_factor=1)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def compute_func():
        a_block = a_cb.wait()
        b_block = b_cb.wait()
        out_block = out_cb.reserve()

        out_block.store(a_block + b_block)

        out_cb.push()
        a_cb.pop()
        b_cb.pop()

    @ttl.datamovement()
    def dm_in():
        a_block = a_cb.reserve()
        tx = ttl.copy(a_in[0, 0], a_block)
        tx.wait()
        a_cb.push()

        b_block = b_cb.reserve()
        tx = ttl.copy(b_in[0, 0], b_block)
        tx.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_out():
        out_block = out_cb.wait()
        tx = ttl.copy(out_block, out[0, 0])
        tx.wait()
        out_cb.pop()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        a_in = ttnn.full((32, 32), 2.0, dtype=ttnn.float32, device=device)
        b_in = ttnn.full((32, 32), 3.0, dtype=ttnn.float32, device=device)
        out = ttnn.empty((32, 32), dtype=ttnn.float32, device=device)

        simple_add(a_in, b_in, out)

        golden = a_in + b_in
        assert_pcc(golden, out)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
