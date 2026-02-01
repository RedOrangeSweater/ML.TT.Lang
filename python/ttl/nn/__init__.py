# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-like modular API: Module, Sequential, Pipeline, Trainer, Pydantic configs.

Use ttl.nn.Sequential or ttl.nn.Pipeline to compose programs; ttl.nn.ProgramModule
wraps a @ttl.program callable. ttl.nn.Trainer provides fit/benchmark/scheduler_env loops
with callbacks and logging. See docs/sdlc/00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md.
"""

from .callbacks import Callback, CallbackAdapter
from .config import PipelineConfig, TrainerConfig
from .device import DeviceManager, TensorAdapter
from .logger import JsonlLogger, Logger, StdoutLogger
from .module import Module, ProgramModule, TrainableModule, TrainableModuleAdapter
from .sequential import Pipeline, Sequential
from .trainer import Trainer

__all__ = [
    "Callback",
    "CallbackAdapter",
    "DeviceManager",
    "JsonlLogger",
    "Logger",
    "Module",
    "Pipeline",
    "PipelineConfig",
    "ProgramModule",
    "Sequential",
    "StdoutLogger",
    "TensorAdapter",
    "Trainer",
    "TrainableModule",
    "TrainableModuleAdapter",
    "TrainerConfig",
]
