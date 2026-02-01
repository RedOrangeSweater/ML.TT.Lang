# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: ttl.nn.Trainer in benchmark mode (compile-only OK).

Uses ProgramModule + Trainer; compile warmup then benchmark loop. When no device
is available, TTLANG_COMPILE_ONLY=1 is used so the example still runs.
"""

import torch

import ttl
import ttl.nn


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


def _make_batch():
    """Single batch: (lhs, rhs, out) as torch tensors. Compile-only path uses these."""
    lhs = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out = torch.zeros((32, 32), dtype=torch.bfloat16)
    return (lhs, rhs, out)


if __name__ == "__main__":
    # Compile-only OK: set TTLANG_COMPILE_ONLY=1 when no hardware.
    batch = _make_batch()

    module = ttl.nn.ProgramModule(simple_add, grid=(1, 1))
    config = ttl.nn.TrainerConfig(
        mode="benchmark",
        max_steps=3,
        compile_warmup_steps=1,
    )
    trainer = ttl.nn.Trainer(model=module, config=config)
    summary = trainer.fit(train_batches=[batch])
    print("Summary:", summary)
    print("Steps:", summary.get("steps"), "metrics:", summary.get("metrics"))
