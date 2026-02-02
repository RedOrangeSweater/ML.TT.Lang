# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-lang settings: environment variables centralized via pydantic-settings.

Replaces scattered os.environ reads in ttl_api.py and diagnostics.py (doc 09).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutoProfileConfig(BaseModel):
    """Configuration for auto-profiling. Validates environment requirements."""

    profile_csv: str | None = None
    tt_metal_home: str | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if not self.profile_csv and not self.tt_metal_home:
            raise ValueError(
                "TTLANG_AUTO_PROFILE=1 requires TT_METAL_HOME or "
                "TTLANG_PROFILE_CSV to be set"
            )
        return self

    def cb_flow_graph_json_path(self) -> str:
        if self.profile_csv:
            return str(Path(self.profile_csv).parent / "cb_flow_graph.json")
        assert self.tt_metal_home is not None
        return f"{self.tt_metal_home}/generated/profiler/.logs/cb_flow_graph.json"


class SettingsTTLang(BaseSettings):
    """Centralized TTLANG_* and related env vars. Defaults preserve existing semantics."""

    model_config = SettingsConfigDict(
        env_prefix="TTLANG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    compile_only: bool = False
    debug_locations: bool = False
    use_scheduler: bool = False
    verbose_errors: bool = False
    verbose_passes: bool = False
    initial_mlir: str | None = None
    final_mlir: str | None = None
    profile_csv: str | None = None

    # Non-TTLANG_ env vars (validation_alias prevents prefix)
    tt_metal_home: str = Field(default="", validation_alias="TT_METAL_HOME")
    user: str = Field(default="default", validation_alias="USER")


# Single instance per process; load at first import of ttl.settings.
settings_ttlang = SettingsTTLang()
