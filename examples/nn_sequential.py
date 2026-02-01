# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: ttl.nn.Sequential (Pipeline) composition.

Build a pipeline from one or more modules; run in order. Here we use
a single ProgramModule inside Sequential to show the API; multi-step
pipelines (output of step i -> first input of step i+1) use the same
Sequential(module0, module1, ...) and call with initial tensors.
"""

import torch
import ttnn

import ttl


@ttl.program(grid=(1, 1))
def simple_add(lhs, rhs, out):
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


if __name__ == "__main__":
    print("=" * 60)
    print("Example: ttl.nn.Sequential (pipeline of modules)")
    print("=" * 60)

    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Sequential by default: one module here; add more for multi-step (output -> next input).
    pipeline = ttl.nn.Sequential(
        ttl.nn.ProgramModule(simple_add, grid=(1, 1)),
    )
    try:
        device = ttnn.open_device(device_id=0)
        lhs_tt = ttnn.from_torch(
            lhs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        rhs_tt = ttnn.from_torch(
            rhs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out_tt = ttnn.from_torch(
            out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        pipeline(lhs_tt, rhs_tt, out_tt)
        out = ttnn.to_torch(out_tt)
        ttnn.close_device(device)
    except Exception:
        import os
        os.environ["TTLANG_COMPILE_ONLY"] = "1"
        pipeline(lhs, rhs, out)
        print("Compile-only: no device, output unchanged.")

    expected = lhs + rhs
    if torch.is_tensor(out) and out.dtype == torch.bfloat16:
        if torch.allclose(out.float(), expected.float(), rtol=1e-2, atol=1e-2):
            print("Output matches expected.")
        else:
            print(f"MISMATCH: max error = {(out.float() - expected.float()).abs().max().item():.6f}")
    else:
        print("Done (compile-only or no device).")
