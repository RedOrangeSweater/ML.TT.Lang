# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Toy example: single-core broadcast using @ttl.program.

Computes bcast(c) + (a * b) in a single fused compute block. One program
compiles to one or more device kernels (here: 1 compute + 2 datamovement).
Same pattern as test/python/simple_bcast.py; decorator is @ttl.program.
"""

import torch
import ttnn

import ttl


@ttl.program(grid=(1, 1))
def fused_bcast(a, b, c, out):
    """bcast(c) + (a * b). Bcast must be first; subsequent ops read from DST."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_cb.wait() as a_tile,
            b_cb.wait() as b_tile,
            c_cb.wait() as c_tile,
            out_cb.reserve() as o,
        ):
            c_bcast = ttl.math.broadcast(c_tile, o, dims=[0])
            ab = a_tile * b_tile
            result = c_bcast + ab
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

        c_blk = c_cb.reserve()
        tx_c = ttl.copy(c[0, 0], c_blk)
        tx_c.wait()
        c_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


if __name__ == "__main__":
    print("=" * 60)
    print("Toy example: single-core broadcast with @ttl.program")
    print("=" * 60)

    a = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    b = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    c = torch.zeros((32, 32), dtype=torch.bfloat16)
    c[0, :] = 1.0
    out = torch.zeros((32, 32), dtype=torch.bfloat16)

    try:
        device = ttnn.open_device(device_id=0)
        a_tt = ttnn.from_torch(
            a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b_tt = ttnn.from_torch(
            b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        c_tt = ttnn.from_torch(
            c,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out_tt = ttnn.from_torch(
            out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        fused_bcast(a_tt, b_tt, c_tt, out_tt)
        out = ttnn.to_torch(out_tt)
        ttnn.close_device(device)
    except Exception:
        import os

        os.environ["TTLANG_COMPILE_ONLY"] = "1"
        fused_bcast(a, b, c, out)
        print("Compile-only: no device, output unchanged")

    expected = (a * b) + c  # 2*3 + 1 = 7.0 in first row
    if torch.is_tensor(out) and out.dtype == torch.bfloat16:
        if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
            print("Output matches expected (a*b + c).")
        else:
            print(
                f"MISMATCH: max error = {(out.float() - expected.float()).abs().max().item():.6f}"
            )
    else:
        print("Done (compile-only or no device).")
