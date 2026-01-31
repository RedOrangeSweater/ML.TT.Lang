# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Opaque type aliases for MLIR objects passed through the compile layer.

We pass the module to ttmlir.passes (get_ttkernel_names, ttkernel_to_cpp_by_name,
get_ttkernel_arg_spec) without inspecting it; typing uses object for opacity.
"""

from __future__ import annotations

# MLIR module from ttmlir.ir (Module.create, etc.). Opaque in our layer.
MlirModuleLike: type[object] = object
