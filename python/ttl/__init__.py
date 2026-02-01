# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Lang Python Package

__version__ = "0.1.0"

# Export TTL DSL API directly at package level so `import ttl; ttl.program` works
# Export generated elementwise operators (auto-generated from TTLElementwiseOps.def)
from ttl._generated_elementwise import *  # noqa: F401,F403
from ttl._generated_elementwise import __all__ as _elementwise_all

# Export additional TTL DSL API classes
from ttl.constants import MemorySpace
from ttl.operators import signpost
from ttl.runtime_tensor import is_ttnn_tensor
from ttl.ttl import (
    Program,
    ProgramSpec,
    compute,
    copy,
    core,
    datamovement,
    grid_size,
    kernel,
    make_circular_buffer_like,
    math,
    program,
    run,
)
from ttl.ttl_api import (
    CircularBuffer,
    CopyTransferHandler,
    TensorBlock,
)

# PyTorch-like modular API (ttl.nn)
from . import nn
from .nn import Module, Pipeline, PipelineConfig, ProgramModule, Sequential

__all__ = [
    "CircularBuffer",
    "CopyTransferHandler",
    "MemorySpace",
    "Program",
    "is_ttnn_tensor",
    "ProgramSpec",
    "TensorBlock",
    "compute",
    "copy",
    "core",
    "datamovement",
    "grid_size",
    "kernel",
    "math",
    "program",
    "run",
    "signpost",
    "make_circular_buffer_like",
    "nn",
    "Module",
    "Pipeline",
    "PipelineConfig",
    "ProgramModule",
    "Sequential",
    # Elementwise operators are automatically included from generated file
    *_elementwise_all,
]
