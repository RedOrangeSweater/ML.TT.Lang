# PR #267: [ttl] Make TRID DMA wait lowering selectable (default: global barriers)

## Problem description

`origin/main` lowers `ttl.copy` and `ttl.wait` with global DMA barriers. That is stable for existing tests, but it does not support TRID-scoped waits needed for issue #87.

This PR adds TRID-aware lowering as an opt-in mode while preserving current default behavior.

## What's changed

### Core lowering (`convert-ttl-to-ttkernel`)

* New pass option `use-trid-barriers` (default `false`):
  * **Default mode**: keeps global `noc_async_{read,write}_barrier()` behavior from `origin/main`
  * **TRID mode**: emits `noc_async_*_set_trid` in copy lowering and `noc_async_*_barrier_with_trid` in wait lowering

* `ttl-to-ttkernel-pipeline` forwards the same option so pass and pipeline behavior are aligned

* Transfer handles are lowered to `i32` TRID values in TRID mode, and SCF structural type conversions are enabled so converted region signatures remain legal

### TRID overflow handling

* `TridAllocator` class tracks outstanding TRIDs (16 slots, 4-bit hardware limit) and their transfer direction
* When a wrapped TRID is still in flight, an automatic matching `barrier_with_trid` is emitted before reuse
* Prevents silent data corruption from TRID reuse conflicts

### TTKernel cleanup patterns

* TRID dedup patterns (`DeduplicateConsecutiveTridBarriers`) are registered only when TRID mode is enabled
* Deduplication for TRID barriers is constrained to identical operands (same TRID/NOC), preserving semantics

### Test coverage

* **Conversion lit tests**:
  * `trid_barriers.mlir` - tests TRID-aware lowering with overflow handling
  * `dma_global_barriers.mlir` - tests default global barrier mode
  * Updated existing tests to use explicit `use-trid-barriers=true` where TRID output is expected

* **Translation lit tests**: updated to enable TRID mode for C++ codegen verification

* **ME2E tests**: parameterized with `use_trid_barriers` option, includes TRID-enabled config

* **Python lit tests**: enabled TRID barriers for hardware execution tests

### Build system fix

* Linux `ttlang-translate` linking now wraps static archives in a linker group (`--start-group`/`--end-group`) to handle circular archive dependencies on ELF linkers

## Commits (6 atomic commits)

1. **[ttl] Add TRID-aware DMA barrier lowering option** - Core implementation
2. **[test] Add lit tests for TRID and global barrier modes** - Conversion tests
3. **[test] Update translate tests for TRID barrier mode** - Translation tests
4. **[test] Parameterize ME2E tests with use_trid_barriers option** - Runtime tests
5. **[test] Enable TRID barriers in Python lit tests** - Hardware tests
6. **[build] Fix ttlang-translate linking on Linux** - Build fix

## Why this design

* Preserves backward-compatible default behavior from `origin/main`
* Enables TRID-aware lowering only when explicitly requested
* Improves correctness and robustness for TRID reuse and rewrite-order-sensitive scenarios
* Keeps optimization and cleanup behavior mode-aware to avoid incorrect cross-mode transformations
* Adds coverage for both default and TRID paths so behavior is explicit and testable

## Validation

```bash
cmake --build build --target check-ttlang
llvm-lit test/ttlang/Conversion/TTLToTTKernel/
llvm-lit test/ttlang/Translate/TTLToCpp/
llvm-lit test/python/
```

## TODOs (future work)

* Profile both modes on representative benchmarks and consider changing the default
* Generalize NOC selection (currently fixed to NOC 0) - see issue #77

Addresses: #87
