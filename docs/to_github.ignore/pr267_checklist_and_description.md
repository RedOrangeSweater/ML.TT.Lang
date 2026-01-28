## PR #267 checklist and description (local helper)

This file is intended as a local helper for creating the GitHub PR description.
It is not required for the build and should not affect CI behavior.

### Context

- **Issue**: tenstorrent/tt-lang issue #87 ("[ttl] Lower ttl.copy ttl.wait to TRID-specific ttkernel noc ops").
- **Reviewer request**: "Implement this as a pass option so we can choose between different lowerings, keeping the default the same as what's in main now."
- **Observed CI failure**: Python lit tests expected `noc_async_{read,write}_barrier()` but codegen produced `noc_async_*_barrier_with_trid(...)`.

### What this PR does (high level)

- Adds a **pass option** to `convert-ttl-to-ttkernel` to select DMA wait lowering:
  - **Default (matches main)**: emit **global** `ttkernel.noc_async_{read,write}_barrier()` and do not emit TRID setup ops.
  - **Opt-in**: emit **TRID-aware** barriers `ttkernel.noc_async_{read,write}_barrier_with_trid(trid, noc)` (and emit `*_set_trid` in the copy lowering) so callers can choose the behavior.
- Plumbs the same option through the `ttl-to-ttkernel-pipeline` for users/tests that use the pipeline driver.
- Updates TRID-focused lit tests to **explicitly enable** TRID mode so the default remains stable and python lit tests keep validating the default.

### Checklist (requirements)

- [ ] **Pass option exists** on `convert-ttl-to-ttkernel` (e.g. `use-trid-barriers`).
- [ ] **Default behavior matches main**:
  - [ ] Default lowering emits `ttkernel.noc_async_read_barrier()` / `ttkernel.noc_async_write_barrier()`.
  - [ ] Default lowering does **not** emit `noc_async_*_set_trid` or `noc_async_*_barrier_with_trid`.
- [ ] **TRID mode is selectable**:
  - [ ] When `use-trid-barriers=1`, lowering emits `noc_async_*_set_trid` and `noc_async_*_barrier_with_trid(...)`.
- [ ] **Pipeline plumbing is present**:
  - [ ] `ttl-to-ttkernel-pipeline` accepts `use-trid-barriers` and forwards it to the pass.
- [ ] **Tests cover both paths appropriately**:
  - [ ] Python lit tests remain unchanged and pass with default lowering.
  - [ ] TRID-specific lit tests explicitly pass `use-trid-barriers=1`.
- [ ] **No scope creep**:
  - [ ] Only pass/pipeline option + related tests are changed (no editor/run-config changes).

### Suggested PR title

`[ttl] Make TRID DMA wait lowering selectable (default: global barriers)`

### Suggested PR description (paste into GitHub)

#### Problem description

The compiler currently emits TRID-aware async DMA barriers (`noc_async_*_barrier_with_trid(...)`) in the default `ttl.wait` lowering. Python lit tests validate the legacy global barrier behavior and therefore fail when the default codegen changes.

Additionally, reviewers requested that this behavior be implemented as a pass option to allow selecting between different lowerings, while keeping the default consistent with `main`.

#### What's changed

- Added a `use-trid-barriers` option to `convert-ttl-to-ttkernel`:
  - Default (`false`): emit global `noc_async_{read,write}_barrier()` and keep python lit expectations stable.
  - Opt-in (`true`): emit `*_set_trid` during copy lowering and `*_barrier_with_trid(trid, noc)` during wait lowering.
- Plumbed `use-trid-barriers` through `ttl-to-ttkernel-pipeline` so the choice is available when running via the pipeline driver.
- Updated TRID-focused MLIR/translation lit tests to explicitly enable TRID mode.

#### Checklist

- [ ] New/Existing tests provide coverage for changes

