# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-like modular API: Module, Sequential, Pipeline, Pydantic configs.

Use ttl.nn.Sequential or ttl.nn.Pipeline to compose programs; ttl.nn.ProgramModule
wraps a @ttl.program callable for use in a pipeline. See docs/sdlc/00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md.
"""

from .config import PipelineConfig
from .module import Module, ProgramModule
from .sequential import Pipeline, Sequential

__all__ = [
    "Module",
    "Pipeline",
    "PipelineConfig",
    "ProgramModule",
    "Sequential",
]
