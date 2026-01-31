# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal FastAPI backend: load scheduler JSON, expose reset/step for UI (doc 16).

Phase 1 (placement): /load, /step, /state. Phase 2 (planned): run dynamics simulator
after placement; return trajectory/metrics; UI displays dynamics and compares plans by metrics.
"""

from __future__ import annotations

from typing import Any, Callable

from scheduler_env import (
    SchedulerPlacementEnv,
    get_action_rcw,
    get_action_random,
)

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Scheduler Viz Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

ALGORITHMS: dict[str, Callable[[SchedulerPlacementEnv], int]] = {
    "rcw": get_action_rcw,
    "random": get_action_random,
}

_env: SchedulerPlacementEnv | None = None
_algorithm_id: str = "rcw"


@app.get("/algorithms")
def algorithms() -> dict[str, list[str]]:
    """Return list of registered algorithm ids (doc 17)."""
    return {"algorithms": list(ALGORITHMS.keys())}


@app.post("/algorithm")
def set_algorithm(algorithm_id: str = Body(..., embed=True)) -> dict[str, str]:
    """Set current algorithm for step_auto (doc 17)."""
    global _algorithm_id
    if algorithm_id not in ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm_id: {algorithm_id}. Use one of {list(ALGORITHMS.keys())}",
        )
    _algorithm_id = algorithm_id
    return {"algorithm_id": _algorithm_id}


@app.post("/load")
def load(body: dict[str, Any]) -> dict[str, Any]:
    """Load scheduler input JSON and reset env. Body may include algorithm_id."""
    global _env, _algorithm_id
    algorithm_id = body.pop("algorithm_id", None)
    if algorithm_id is not None and algorithm_id not in ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm_id: {algorithm_id}. Use one of {list(ALGORITHMS.keys())}",
        )
    if algorithm_id is not None:
        _algorithm_id = algorithm_id
    try:
        _env = SchedulerPlacementEnv(body)
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
        "algorithm_id": _algorithm_id,
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


@app.post("/step_auto")
def step_auto() -> dict[str, Any]:
    """Get action from current algorithm and execute one step (doc 17)."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /load first")
    policy = ALGORITHMS[_algorithm_id]
    action = policy(_env)
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
        "action": action,
        "algorithm_id": _algorithm_id,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    """Return current full state for UI (includes algorithm_id in session)."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /load first")
    out = _env.get_full_state()
    out["algorithm_id"] = _algorithm_id
    return out
