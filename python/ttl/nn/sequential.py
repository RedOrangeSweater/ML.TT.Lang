# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sequential and Pipeline: composition of modules (run in order, output -> next input)."""

from __future__ import annotations

from typing import Sequence

from .module import Module, ProgramModule


class Sequential:
    """Composition of modules: runs in order; output of step i is first input of step i+1.

    Each step is a Module (ProgramModule or custom). Call with initial tensors;
    step 0 gets all args; step i (i > 0) gets (output of step i-1,) as first arg.
    compile() runs the full sequence once to warmup compile caches.
    """

    def __init__(self, *modules: Module | ProgramModule) -> None:
        """Build Sequential from a sequence of modules.

        Args:
            *modules: Module instances (ProgramModule or any callable matching Module protocol).
        """
        self._modules: list[Module] = list(modules)

    def compile(self, *args: object, **kwargs: object) -> None:
        """Run the full sequence once to warmup compile caches. Same as one __call__ for side effect."""
        self(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> object | None:
        """Run modules in sequence; output of each step is first input of next."""
        if not self._modules:
            return None
        out = self._modules[0](*args, **kwargs)
        for mod in self._modules[1:]:
            if out is None:
                return None
            out = mod(out, **kwargs)
        return out


# Alias for pipeline-style usage (same behavior as Sequential).
Pipeline = Sequential
