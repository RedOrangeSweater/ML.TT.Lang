#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Compare current tree to baseline: run same test subset and report pass/fail.
# Run from repo root with build env activated.
# Reads baseline_commit.txt and baseline_result.txt from scripts/ or BASELINE_DIR.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_DIR="${BASELINE_DIR:-$SCRIPT_DIR}"
COMMIT_FILE="$BASELINE_DIR/baseline_commit.txt"
RESULT_FILE="$BASELINE_DIR/baseline_result.txt"

if [[ ! -f "$COMMIT_FILE" ]]; then
  echo "Baseline not found: $COMMIT_FILE. Run baseline_record.sh first."
  exit 1
fi

BASELINE_COMMIT=$(cat "$COMMIT_FILE")
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Baseline commit: $BASELINE_COMMIT"
echo "Current commit:  $CURRENT_COMMIT"

echo "Running test subset (check-ttlang-pytest)..."
if cmake --build build --target check-ttlang-pytest; then
  echo "Result: PASS (no regression vs baseline)"
else
  echo "Result: FAIL (regression)"
  exit 1
fi
