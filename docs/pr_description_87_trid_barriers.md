# PR Description
### Title

`[TTL/TTKernel] Lower ttl.copy/ttl.wait to TRID-specific NOC barriers`

Fixes #87

---

### What?

This PR replaces *global* NOC async barriers produced from `ttl.wait` with **TRID-specific** barriers so each wait synchronizes only the async transfer it is associated with.

Concretely:

- `ttl.copy` allocates a per-copy TRID, emits `ttkernel.noc_async_{read,write}_set_trid(trid, noc=0)`, and returns the TRID as an `i32` SSA value.
- `ttl.wait` lowers to `ttkernel.noc_async_{read,write}_barrier_with_trid(trid, noc=0)` instead of the global `ttkernel.noc_async_{read,write}_barrier()`.
- TTKernel cleanup is extended to deduplicate consecutive TRID barriers **only** when operands match (same `trid` and `noc`).
- TTL->TTKernel conversion tests are updated, and a focused TRID regression test is added.
- **CI follow-up**: TTL->C++ translation lit tests are updated to expect `noc_async_{read,write}_set_trid(...)` and `noc_async_{read,write}_barrier_with_trid(...)`, and the TTL->TTKernel conversion adds SCF structural type conversions so loop region signatures stay legal when handle types change.

---

### Why?

Global barriers (`noc_async_*_barrier()`) wait for *all* outstanding async transactions of a direction. This is overly conservative and prevents overlap of independent DMA operations.

Using TRID-specific barriers enables:

- correct synchronization at the granularity of an individual transaction,
- improved potential concurrency/overlap (independent transfers do not block each other),
- more predictable correctness when multiple async DMAs are in flight.

---

### How?

#### 1) TRID flows through SSA (handle -> `i32`)

In the TTL->TTKernel type converter:

- `!ttl.transfer_handle<read|write>` is converted to **`i32`**, representing the transaction id (TRID) as an SSA value.
- For non-TTNN-layout ranked tensors, element types are converted so that tensors containing handles can be expressed as tensors containing `i32` where this is legal.

Direction (read/write) is still determined from the original `ttl.wait` operand type.

#### 2) `ttl.copy` assigns and sets TRID

In `ConvertTTLToTTKernel.cpp` lowering:

- A per-pass `TridAllocator` assigns `trid = next++ & 0xF` (deterministic, simple).
- For tensor-slice -> CB reads:
  - `ttkernel.noc_async_read_set_trid(trid, noc=0)` is emitted once before the tile loop.
- For CB -> tensor-slice writes:
  - `ttkernel.noc_async_write_set_trid(trid, noc=0)` is emitted once before the tile loop.
- The `ttl.copy` result is replaced by the `i32` TRID SSA value.

#### 3) `ttl.wait` becomes TRID-specific barrier

`ttl.wait` now lowers to:

- `ttkernel.noc_async_read_barrier_with_trid(trid, noc=0)` for read handles
- `ttkernel.noc_async_write_barrier_with_trid(trid, noc=0)` for write handles

This ensures no global barriers are emitted for `ttl.wait`.

#### 4) Cleanup: safe dedup for TRID barriers

`TTKernelCleanupPatterns.cpp` adds a rewrite that removes consecutive
`noc_async_{read,write}_barrier_with_trid` only when the operands match, avoiding
incorrectly deduplicating waits on different TRIDs.

#### 5) Structural type conversions for loops (CI follow-up)

Because transfer handles become `i32` TRIDs, `scf.for` iter_args/results that carry
handles also change type. The TTL->TTKernel conversion now populates SCF structural
type conversion patterns/legality so SCF region signatures are updated consistently.

#### 6) Tests

TTL->TTKernel conversion tests:

- Updated: `test/ttlang/Conversion/TTLToTTKernel/dma_single_core.mlir`
- Updated: `test/ttlang/Conversion/TTLToTTKernel/loopback_dram_copy.mlir`
- Added: `test/ttlang/Conversion/TTLToTTKernel/trid_barriers.mlir`

TTL->C++ translation tests (CI follow-up):

- Updated: `test/ttlang/Translate/TTLToCpp/*` cases to check for `*_set_trid(...)` and `*_barrier_with_trid(...)` instead of `*_barrier()`.

The `trid_barriers.mlir` test covers:

- single copy+wait -> `*_set_trid` + `*_barrier_with_trid` and **no** global barriers
- two copy+wait pairs -> distinct TRIDs and matching waits

---

### How to Test?

From the `tt-lang` repo root:

```bash
source build/env/activate

# Build the tool (if needed)
cmake --build build --target ttlang-opt

# Run the conversion lit suite
llvm-lit -sv test/ttlang/Conversion/TTLToTTKernel
```

---

### Checklist

- [x] Self-reviewed (style, logic)
- [x] Added/updated tests
- [x] PR is small and focused on one task (#87)

