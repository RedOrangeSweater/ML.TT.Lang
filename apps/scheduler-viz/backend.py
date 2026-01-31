# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal FastAPI backend: load scheduler JSON, expose reset/step for UI (doc 16).

Phase 1 (placement): /load, /step, /state. Phase 2 (planned): run dynamics simulator
after placement; return trajectory/metrics; UI displays dynamics and compares plans by metrics.
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from scheduler_env import SchedulerPlacementEnv

app = FastAPI(title="Scheduler Viz Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

_env: SchedulerPlacementEnv | None = None


@app.post("/load")
def load(data: dict[str, Any]) -> dict[str, Any]:
    """Load scheduler input JSON and reset env. Returns initial state."""
    global _env
    try:
        _env = SchedulerPlacementEnv(data)
        obs, info = _env.reset(seed=42)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": {
            "placement": obs["placement"].tolist(),
            "next_node": int(obs["next_node"]),
        },
        "info": {k: v for k, v in info.items() if k != "graph" and k != "topology"},
        "full_state": _env.get_full_state(),
        "action_space_n": _env.action_space.n,
    }


@app.post("/step")
def step(action: int = Body(..., embed=True)) -> dict[str, Any]:
    """Execute one step. Returns observation, reward, terminated, info, full_state."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /load first")
    obs, reward, terminated, truncated, info = _env.step(action)
    return {
        "observation": {
            "placement": obs["placement"].tolist(),
            "next_node": int(obs["next_node"]),
        },
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "full_state": _env.get_full_state(),
    }


@app.get("/state")
def state() -> dict[str, Any]:
    """Return current full state for UI."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /load first")
    return _env.get_full_state()
