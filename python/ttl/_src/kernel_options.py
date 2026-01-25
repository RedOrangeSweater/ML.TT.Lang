"""Pydantic models for tt-lang Python decorator options."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..constants import SUPPORTED_MEMORY_SPACES


class KernelDecoratorOptions(BaseModel):
    """Validated options for the pykernel_gen decorator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Sequence[int] | Callable[..., Sequence[int]]
    indexing_maps: list[Callable[..., object]] = Field(default_factory=list)
    iterator_types: list[str] = Field(default_factory=list)
    num_outs: int = 1
    memory_space: str = "L1"
    tiled: bool = True

    @field_validator("grid", mode="before")
    @classmethod
    def _require_grid(cls, value: object) -> object:
        if value is None:
            raise ValueError("grid parameter is required")
        if callable(value):
            return value
        if isinstance(value, Sequence):
            return tuple(value)
        raise TypeError(
            "grid must be a sequence of ints or a callable returning a sequence"
        )

    @field_validator("indexing_maps", "iterator_types", mode="before")
    @classmethod
    def _none_to_list(cls, value: object) -> object:
        if value is None:
            return []
        return value

    @field_validator("num_outs")
    @classmethod
    def _validate_num_outs(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"num_outs must be 1, got {value}")
        return value

    @field_validator("memory_space")
    @classmethod
    def _validate_memory_space(cls, value: str) -> str:
        if value not in SUPPORTED_MEMORY_SPACES:
            supported = ", ".join(sorted(SUPPORTED_MEMORY_SPACES))
            raise ValueError(f"Invalid memory_space: {value!r}. Must be one of: {supported}")
        return value

    @field_validator("tiled")
    @classmethod
    def _validate_tiled(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"tiled must be a boolean, got {type(value).__name__}")
        return value

    @model_validator(mode="after")
    def _validate_indexing(self) -> "KernelDecoratorOptions":
        if self.iterator_types and not self.indexing_maps:
            raise ValueError("indexing_maps must be set when iterator_types is set")
        if self.indexing_maps and self.iterator_types:
            expected = len(self.iterator_types)
            for indexing_map in self.indexing_maps:
                param_names = list(inspect.signature(indexing_map).parameters)
                if len(param_names) != expected:
                    raise ValueError(
                        "Number of dimensions "
                        f"({len(param_names)}) must match iterator_types length ({expected})"
                    )
        return self
