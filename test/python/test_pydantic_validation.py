# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight Pydantic validation tests (no device / full ttl import)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

# Stub ttl package so scheduler.config loads without compiled ttl extensions.
_ttl_root = _PYTHON / "ttl"
if "ttl" not in sys.modules:
    _ttl = types.ModuleType("ttl")
    _ttl.__path__ = [str(_ttl_root)]
    sys.modules["ttl"] = _ttl
if "ttl.scheduler" not in sys.modules:
    _sched = types.ModuleType("ttl.scheduler")
    _sched.__path__ = [str(_ttl_root / "scheduler")]
    sys.modules["ttl.scheduler"] = _sched

from ttl.scheduler.config import AbstractEngineConfig, load_abstract_engine_config


def test_invalid_backend_rejected():
    with pytest.raises(ValidationError):
        AbstractEngineConfig.model_validate({"backend": "invalid_backend"})


def test_disconnected_topology_rejected():
    with pytest.raises(ValidationError, match="not connected"):
        AbstractEngineConfig(
            backend="toy_shops",
            topology_graph={
                "nodes": ["a", "b"],
                "edges": [],
            },
        )


def test_settings_json_roundtrip(tmp_path: Path):
    cfg = AbstractEngineConfig(
        backend="tenstorrent",
        topology_grid={"grid_cols": 2, "grid_rows": 2},
    )
    path = tmp_path / "engine.json"
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    loaded = load_abstract_engine_config(path)
    assert loaded.backend == "tenstorrent"
    assert loaded.get_topology_grid() == (2, 2)


def test_example_config_file_loads():
    path = Path(__file__).parent / "example_abstract_engine_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = AbstractEngineConfig.model_validate(raw)
    assert cfg.backend == "toy_shops"
    assert cfg.graph_toy_generator is not None
