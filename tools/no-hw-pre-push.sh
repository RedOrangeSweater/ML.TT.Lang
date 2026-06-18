#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run CI-equivalent checks without Tenstorrent hardware (fork debug / pre-push).
#
# Usage (from repo root):
#   ./tools/no-hw-pre-push.sh
#   ./tools/no-hw-pre-push.sh --in-docker    # already inside IRD container with build/
#
# Default: runs checks in ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04:v1.1.3

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${TTLANG_NOHW_IMAGE:-ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04:v1.1.3}"
MEMORY="${TTLANG_NOHW_MEMORY:-18g}"

run_checks() {
  set -euo pipefail
  cd /work

  if [[ ! -d build/env ]]; then
    echo "Configuring tt-lang (TTLANG_USE_TOOLCHAIN=ON)..."
    cmake -G Ninja -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTTLANG_USE_TOOLCHAIN=ON
    cmake --build build
  fi

  # shellcheck disable=SC1091
  source build/env/activate

  if command -v pre-commit >/dev/null 2>&1; then
    pre-commit run --all-files
  else
    echo "pre-commit not installed; skipping (CI will run it)"
  fi

  cmake --build build --target check-ttlang-mlir
  cmake --build build --target check-ttlang-python-bindings
  cmake --build build --target check-ttlang-python-lit
  python test/python/smoketest.py

  PYTHONPATH=python:examples pytest test/sim/ -v --tb=short -q

  if [[ -x test/scripts/tt-lang-sim-pytest ]]; then
    python test/scripts/tt-lang-sim-pytest test/python/simple_add.py -v
  fi

  echo "NO-HW GATE OK"
}

if [[ "${1:-}" == "--in-docker" ]]; then
  run_checks
  exit 0
fi

docker run --rm \
  --memory="${MEMORY}" --memory-swap="${MEMORY}" --cpus=6 \
  -e CMAKE_BUILD_PARALLEL_LEVEL=2 \
  -v "${ROOT}:/work" \
  -w /work \
  "${DOCKER_IMAGE}" \
  bash -lc './tools/no-hw-pre-push.sh --in-docker'
