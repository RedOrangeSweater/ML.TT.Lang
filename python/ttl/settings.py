"""
Typed configuration for the TTL Python frontend.

This module centralizes all TTLANG_* environment-driven configuration using
`pydantic-settings`. Code should import the module-level `settings` object and
read typed fields, rather than reaching into `os.environ` directly.
"""

from __future__ import annotations

from pathlib import Path

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


settings = TTLangSettings()


def reload_settings() -> TTLangSettings:
    """Reload settings from the environment.

    Intended for tests/REPL; production code should read `settings`.
    """

    global settings
    settings = TTLangSettings()
    return settings

