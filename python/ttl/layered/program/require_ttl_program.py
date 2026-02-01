# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..context import RunContext


def require_ttl_program_attr(attr_name: str):
    """Ensure ctx.req.spec.program has a marker attribute (e.g. _ttl_program)."""

    def _mw(ctx: RunContext) -> RunContext:
        req = ctx.req
        if req is None:
            raise RuntimeError("require_ttl_program_attr requires ctx.req to be set")
        if not getattr(req.spec.program, attr_name, False):
            raise NotImplementedError(
                "run() with a raw callable (e.g. lambda) is not yet implemented: "
                "inference from parameters and return type is planned. "
                "Use @ttl.program to define the kernel and pass "
                "RunRequest.from_program(program, *args, grid=...). "
                "See docs/sdlc/00_Main/02_Architecture/"
                "20_IdealDataFlowAndModuleStructure.md "
                "(lambda + inference)."
            )
        return ctx

    return _mw


__all__ = ["require_ttl_program_attr"]
