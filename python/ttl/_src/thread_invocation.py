"""Helpers for building internal kwargs for thread compilation wrappers.

This module centralizes the internal keyword arguments injected by the
`@ttl.compute()` / `@ttl.datamovement()` decorators so callers do not
manually assemble `_source_file`, `_source_lines`, etc.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ThreadCompilerKwargs(BaseModel):
    """Validated internal kwargs for `TTLGenericCompiler` invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_file: str = Field(alias="_source_file")
    source_lines: list[str] = Field(alias="_source_lines")
    line_offset: int = Field(alias="_line_offset")
    globals: dict[str, object] = Field(alias="_globals")

    debug_locations: bool = True

    verbose: bool = Field(default=False, alias="_verbose")
    source_code: list[str] | None = Field(default=None, alias="_source_code")

    @model_validator(mode="after")
    def _validate_verbose_source_code(self) -> "ThreadCompilerKwargs":
        if self.verbose and self.source_code is None:
            raise ValueError("_source_code must be set when _verbose is true")
        if (not self.verbose) and self.source_code is not None:
            raise ValueError("_source_code must be omitted when _verbose is false")
        return self


def build_thread_compiler_kwargs(
    thread_fn: Callable[..., object],
    *,
    source_file: str,
    source_lines: list[str],
    line_offset: int,
    verbose: bool,
) -> dict[str, object]:
    """Build validated internal kwargs for compiling a thread function.

    Notes:
    - Heavy work (source extraction, AST parsing) should happen outside models.
    - This function only validates and maps internal keys.
    """

    model = ThreadCompilerKwargs(
        _source_file=source_file,
        _source_lines=source_lines,
        _line_offset=line_offset,
        _globals=getattr(thread_fn, "__globals__", {}),
        _verbose=verbose,
        _source_code=source_lines if verbose else None,
        debug_locations=True,
    )
    return model.model_dump(by_alias=True, exclude_none=True)

