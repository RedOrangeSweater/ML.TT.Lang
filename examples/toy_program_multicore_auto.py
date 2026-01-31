# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Toy example: multicore elementwise with @ttl.program(grid="auto").

Computes y = a * b + c over a 2D grid; grid size is inferred from tensor
shapes. One program compiles to one or more device kernels. Same pattern
as examples/tutorial/multicore_grid_auto.py; decorator is @ttl.program.
"""

import torch
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 4


@ttl.program(grid="auto")
def fused_mul_add(a, b, c, y):
    """y = a * b + c per tile block; grid auto from shapes."""
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    grid_cols, grid_rows = ttl.grid_size(dims=2)

    rows = a.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = a.shape[1] // TILE_SIZE // col_tiles_per_block

    rows_per_core = -(-rows // grid_rows)
    cols_per_core = -(-cols // grid_cols)

    a_cb = ttl.make_circular_buffer_like(
        a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    c_cb = ttl.make_circular_buffer_like(
        c, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    y_cb = ttl.make_circular_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )

    @ttl.compute()
    def demo_compute():
        core_col, core_row = ttl.core(dims=2)

        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        with (
                            a_cb.wait() as a_blk,
                            b_cb.wait() as b_blk,
                            c_cb.wait() as c_blk,
                            y_cb.reserve() as y_blk,
                        ):
                            y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def demo_read():
        core_col, core_row = ttl.core(dims=2)

        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                start_row_tile = row * row_tiles_per_block
                end_row_tile = (row + 1) * row_tiles_per_block

                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        start_col_tile = col * col_tiles_per_block
                        end_col_tile = (col + 1) * col_tiles_per_block

                        with (
                            a_cb.reserve() as a_blk,
                            b_cb.reserve() as b_blk,
                            c_cb.reserve() as c_blk,
                        ):
                            tx_a = ttl.copy(
                                a[
                                    start_row_tile:end_row_tile,
                                    start_col_tile:end_col_tile,
                                ],
                                a_blk,
                            )
                            tx_b = ttl.copy(
                                b[
                                    start_row_tile:end_row_tile,
                                    start_col_tile:end_col_tile,
                                ],
                                b_blk,
                            )
                            tx_c = ttl.copy(
                                c[
                                    start_row_tile:end_row_tile,
                                    start_col_tile:end_col_tile,
                                ],
                                c_blk,
                            )

                            tx_a.wait()
                            tx_b.wait()
                            tx_c.wait()

    @ttl.datamovement()
    def demo_write():
        core_col, core_row = ttl.core(dims=2)

        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                start_row_tile = row * row_tiles_per_block
                end_row_tile = (row + 1) * row_tiles_per_block

                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        start_col_tile = col * col_tiles_per_block
                        end_col_tile = (col + 1) * col_tiles_per_block

                        with y_cb.wait() as y_blk:
                            tx = ttl.copy(
                                y_blk,
                                y[
                                    start_row_tile:end_row_tile,
                                    start_col_tile:end_col_tile,
                                ],
                            )
                            tx.wait()


if __name__ == "__main__":
    print("=" * 60)
    print("Toy example: multicore grid=auto with @ttl.program")
    print("=" * 60)

    shape = (256, 256)
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand(shape, dtype=torch.bfloat16)
    y = torch.zeros(shape, dtype=torch.bfloat16)

    try:
        device = ttnn.open_device(device_id=0)
        a_tt = ttnn.from_torch(
            a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b_tt = ttnn.from_torch(
            b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        c_tt = ttnn.from_torch(
            c,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y_tt = ttnn.from_torch(
            y,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_mul_add(a_tt, b_tt, c_tt, y_tt)
        y = ttnn.to_torch(y_tt)
        ttnn.close_device(device)

        expected = a * b + c
        if torch.allclose(y, expected, rtol=1e-2, atol=1e-2):
            print("Output matches expected (a*b + c).")
        else:
            err = (y.float() - expected.float()).abs().max().item()
            print(f"MISMATCH: max error = {err:.6f}")
    except Exception:
        import os

        os.environ["TTLANG_COMPILE_ONLY"] = "1"
        fused_mul_add(a, b, c, y)
        print("Compile-only: no device, output unchanged.")
