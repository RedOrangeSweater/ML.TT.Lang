#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Record baseline: current git commit hash and result of test subset.
# Run from repo root with build env activated (e.g. source build/env/activate).
# Output: baseline_commit.txt, baseline_result.txt in scripts/ or current dir.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_DIR="${BASELINE_DIR:-$SCRIPT_DIR}"
COMMIT_FILE="$BASELINE_DIR/baseline_commit.txt"
RESULT_FILE="$BASELINE_DIR/baseline_result.txt"

git rev-parse HEAD > "$COMMIT_FILE"
echo "Recorded baseline commit: $(cat "$COMMIT_FILE")"

echo "Running test subset (check-ttlang-pytest)..."
if cmake --build build --target check-ttlang-pytest > /tmp/baseline_pytest.log 2>&1; then
  echo "PASS" > "$RESULT_FILE"
  echo "Pytest: PASS"
else
  echo "FAIL" > "$RESULT_FILE"
  echo "Pytest: FAIL (see /tmp/baseline_pytest.log)"
  exit 1
fi
