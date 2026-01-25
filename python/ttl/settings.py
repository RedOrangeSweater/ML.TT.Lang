"""Environment-backed settings for the tt-lang Python API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TTLangSettings(BaseSettings):
    """Typed environment settings for tt-lang runtime and diagnostics."""

    model_config = SettingsConfigDict(env_prefix="TTLANG_", extra="ignore")

    compile_only: bool = False
    debug_locations: bool = False
    initial_mlir: Path | None = None
    final_mlir: Path | None = None
    verbose_passes: bool = False
    verbose_errors: bool = False
    profile_compile: bool = False
    profile_compile_out: Path | None = None
    tracy: bool = False

    @field_validator(
        "initial_mlir",
        "final_mlir",
        "profile_compile_out",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> TTLangSettings:
    """Return cached settings loaded from environment variables."""

    return TTLangSettings()
