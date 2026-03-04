# PR #267 — Review comments addressed (local reference)

Файл для копирования ответов в GitHub UI. Не публиковать автоматически.

## Новые пофикшены (эта сессия)

Комменты от 2026-03-02, закрыты в текущей сессии:

- **4** — удалён неиспользуемый `releaseTrid`
- **5** — `cb_to_tensor_single_tile_write.mlir`: два RUN (default + TRID), раздельные CHECK/TRID
- **6** — убран `allocateTrid` в ветке без TRID
- **7** — проверка аллокатора заменена на `assert`
- **8** — C-style arrays → SmallVector, единый стиль имён (`_` у приватных)
- **9** — хелпер `emitNocBarrier`, убрано дублирование веток
- **10** — проверка типа handle в WaitLowering заменена на `assert`
- **11** — в config_specs добавлен multi-tile конфиг с `use_trid_barriers=True`

**1–3** — [ANSWERED], ответы в PR уже есть.

---

**[ANSWERED] 1. lib/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.cpp**  
*Comment:* Probably doesn't matter that much, but could make the relevant patterns conditional on the option that enables TRID?

**fixed:**
- `populateTTKernelCleanupPatterns` now takes `useTridBarriers` (default false in header); TRID dedup patterns are only added when true.
- Convert pass calls it with `useTridBarriers` so the option is forwarded to cleanup.

---

**[ANSWERED] 2. include/ttlang/Dialect/TTL/Passes.td (line 31)**  
*Comment:* Thank you for adding the option! … it would be interesting to profile the different approaches … add a short TODO to that effect here if you agree?

**fixed:**
- Added TODO in pass description: “Profile both modes on representative benchmarks and consider changing the default.”

---

**[ANSWERED] 3. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 594) — TRID overflow**  
*Comment:* There is wrapping at 16 TRIDs, but what happens if the 0th, etc are still not completed at that point? … Maybe add a TODO for future improvement.

**fixed:**
- `TridAllocator` now tracks outstanding TRIDs and their direction; when a TRID would be reused while still in-flight, CopyLowering emits a matching `barrier_with_trid` before reassigning.
- WaitLowering releases TRIDs via `releaseTrid()` so they can be reused without an auto-barrier.
- Lit test (17 copies, no intervening waits) verifies auto-barrier on overflow.  
*(Reviewer replied: “Nice!”)*

---

**4. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 588)** **[NEW]**  
*Comment:* Is it used anywhere? If not, remove.

**fixed:**
- `releaseTrid` was never called; removed the method.

---

**5. test/ttlang/Translate/TTLToCpp/cb_to_tensor_single_tile_write.mlir** **[NEW]**  
*Comment:* I would not change the RUN command for existing tests, that removes coverage for the default lowering path; instead, add a new RUN line with a different check prefix, e.g. TRID, and add the new TRID: lines where output differs from the default CHECK lines.

**fixed:**
- First RUN: default pipeline (no `use-trid-barriers`), CHECK for global barriers, CHECK-NOT for set_trid/barrier_with_trid.
- Second RUN: `use-trid-barriers=1`, output to `%t.trid.cpp`, FileCheck with prefix TRID for TRID-specific lines.
- Default path and TRID path are both covered.

---

**6. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 795)** **[NEW]**  
*Comment:* Why is this necessary if TRIDs are not being used?

**fixed:**
- In global-barrier mode TRID tracking is not needed; removed `tridAllocator->allocateTrid(direction)` from the else branch.
- Comment added: “Global-barrier mode: no TRID tracking; handle is always constant 0.”

---

**7. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 767)** **[NEW]**  
*Comment:* nit: dead code — the allocator currently is always constructed; I think an assert is more appropriate here or llvm_unreachable.

**fixed:**
- Replaced `if (!tridAllocator) return rewriter.notifyMatchFailure(...)` with `assert(tridAllocator && "CopyLowering requires TRID allocator")`.

---

**8. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 593)** **[NEW]**  
*Comment:* Any reason for using C-style arrays vs SmallVector? nit: inconsistent naming of private vars w.r.t. trailing underscore.

**fixed:**
- Replaced C-style arrays with `llvm::SmallVector<bool, kNumTrids> outstanding_` and `llvm::SmallVector<TransferKind, kNumTrids> direction_` (initialized to size kNumTrids).
- Renamed `nextTrid` to `nextTrid_` for consistency with existing `direction_`.

---

**9. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 674)** **[NEW]**  
*Comment:* Minor: Both TRID and non-TRID branches duplicate the same if (read) / else if (write) / else chain. Consider creating a helper to emit the proper kind of barrier without replicating logic.

**fixed:**
- Added `emitNocBarrier(rewriter, loc, kind, tridVal, useTridBarriers)` that emits either TRID-scoped or global barrier.
- CopyLowering (evict path) and WaitLowering both use this helper; duplicate if/else chains removed.

---

**10. lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp (line 849)** **[NEW]**  
*Comment:* I think this should be an assert — the type converter ensures i32 already.

**fixed:**
- Replaced the `if (!tridVal.getType().isInteger(32)) return notifyMatchFailure(...)` with `assert(tridVal.getType().isInteger(32) && "transfer handle must be type-converted to i32 before ttl.wait")`.

---

**11. test/me2e/config_specs.py (line 147)** **[NEW]**  
*Comment:* Should probably also add multi-tile config with `use_trid_barriers=True`?

**fixed:**
- Added `TestConfig(num_tiles=4, block_h=2, block_w=2, use_trid_barriers=True)` to CONFIGS.
