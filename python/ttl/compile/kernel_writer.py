# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kernel writer: write compiled kernel source to temp directory.

Single responsibility: KernelWriteRequest -> file path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..descriptor_options import KernelWriteRequest
from ..settings import settings_ttlang


def write_kernel_to_tmp(req: KernelWriteRequest) -> str:
    """Write kernel source to req.base_dir or /tmp/{user} and return the file path."""
    content_hash = hashlib.md5(req.source.encode()).hexdigest()[:8]
    if req.base_dir is not None:
        req.base_dir.mkdir(parents=True, exist_ok=True)
        path = req.base_dir / f"ttlang_kernel_{req.name}_{content_hash}.cpp"
    else:
        user = settings_ttlang.user
        path = Path(f"/tmp/{user}/ttlang_kernel_{req.name}_{content_hash}.cpp")
        os.makedirs(f"/tmp/{user}", exist_ok=True)
    with path.open("w") as f:
        f.write(req.source)
    print(f"=== {req.name} kernel written to {path} ===")
    print(req.source)
    print("=" * 60)
    return str(path)


__all__ = ["write_kernel_to_tmp"]
