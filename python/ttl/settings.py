"""
Typed configuration for the TTL Python frontend.

This module centralizes all TTLANG_* environment-driven configuration using
`pydantic-settings`. Code should import the module-level `settings` object and
read typed fields, rather than reaching into `os.environ` directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TTLangSettings(BaseSettings):
    """Environment-backed settings for the TTL Python frontend."""

    model_config = SettingsConfigDict(
        env_prefix="TTLANG_",
        extra="ignore",
    )

    # If true, compile but do not execute (e.g. for CI or IR inspection).
    compile_only: bool = False

    # If true, print locations in MLIR output. Locations are still generated
    # regardless to improve error messages.
    debug_locations: bool = False

    # If set, save MLIR snapshots to these paths.
    initial_mlir_path: Path | None = None
    final_mlir_path: Path | None = None

    # Enable pass manager IR printing.
    verbose_passes: bool = False

    # Enable more verbose MLIR error formatting.
    verbose_errors: bool = False

    @field_validator(
        "compile_only",
        "debug_locations",
        "verbose_errors",
        mode="before",
    )
    @classmethod
    def _parse_bool_one_is_true(cls, v: object) -> bool:
        # Preserve existing semantics where these toggles were enabled only by "1".
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v == "1"
        return bool(v)

    @field_validator("verbose_passes", mode="before")
    @classmethod
    def _parse_verbose_passes(cls, v: object) -> bool:
        # Preserve existing semantics where presence (non-empty) enabled this toggle.
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v != ""
        return bool(v)

    @field_validator("initial_mlir_path", "final_mlir_path", mode="before")
    @classmethod
    def _parse_path_or_none(cls, v: object) -> Path | None:
        if v is None:
            return None
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            if v == "":
                return None
            return Path(v)
        return Path(str(v))


settings = TTLangSettings()


def reload_settings() -> TTLangSettings:
    """Reload settings from the environment.

    Intended for tests/REPL; production code should read `settings`.
    """

    global settings
    settings = TTLangSettings()
    return settings

