# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal FastAPI backend: load scheduler JSON, expose reset/step for UI (doc 16).

Phase 1 (placement): /load, /step, /state. Thin endpoints; delegation to services.
Uses Pydantic models, algorithm registry, and a single session object.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from algorithms import ALGORITHMS, UnknownAlgorithmError
from models import (
    BackendSession,
    LoadExampleRequest,
    LoadResponse,
    SetAlgorithmRequest,
    StateResponse,
    StepRequest,
    StepResponse,
    SchedulerInputPayload,
    observation_from_env,
)
from scheduler_env import SchedulerPlacementEnv

app = FastAPI(title="Scheduler Viz Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _list_examples() -> list[str]:
    """Return example names from fixtures/*.json (stem only)."""
    if not FIXTURES_DIR.exists():
        return []
    return sorted(p.stem for p in FIXTURES_DIR.glob("*.json"))


def _get_session(request: Request) -> BackendSession:
    """Return current session or raise 400 if not loaded."""
    session = getattr(request.app.state, "session", None)
    if session is None:
        raise HTTPException(status_code=400, detail="Call /load first")
    return session


def _build_load_response(session: BackendSession) -> dict[str, Any]:
    """Build response after load/load_example from session."""
    obs = session.env._obs()
    return LoadResponse(
        observation=observation_from_env(obs["placement"], obs["next_node"]),
        info={},
        full_state=session.env.get_full_state(),
        action_space_n=session.env.action_space.n,
        algorithm_id=session.algorithm_id,
        reference_reward=session.reference_reward,
    ).model_dump()


def _build_step_response(
    session: BackendSession,
    obs: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    action: int | None = None,
) -> dict[str, Any]:
    """Build response after step/step_auto."""
    payload = StepResponse(
        observation=observation_from_env(obs["placement"], obs["next_node"]),
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=info,
        full_state=session.env.get_full_state(),
        action=action,
        algorithm_id=session.algorithm_id,
        reference_reward=session.reference_reward,
    )
    return payload.model_dump(exclude_none=False)


def _build_state_response(session: BackendSession) -> dict[str, Any]:
    """Build /state response."""
    return StateResponse(
        full_state=session.env.get_full_state(),
        algorithm_id=session.algorithm_id,
        reference_reward=session.reference_reward,
    ).model_dump()


def run_until_done(env: SchedulerPlacementEnv, policy: Any) -> float:
    """Run env with policy until terminated or truncated; return total reward."""
    total_reward = 0.0
    while True:
        action = policy(env)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


@contextmanager
def session_scope() -> Generator[None, None, None]:
    """Context manager: clear app.state.session on exit (for tests or scripts)."""
    try:
        yield
    finally:
        if hasattr(app.state, "session"):
            app.state.session = None


@app.get("/examples")
def examples() -> dict[str, list[str]]:
    """Return list of example names (from fixtures/*.json)."""
    return {"examples": _list_examples()}


@app.get("/algorithms")
def algorithms() -> dict[str, list[str]]:
    """Return list of registered algorithm ids (doc 17)."""
    return {"algorithms": ALGORITHMS.ids()}


@app.post("/algorithm")
def set_algorithm(body: SetAlgorithmRequest, request: Request) -> dict[str, str]:
    """Set current algorithm for step_auto (doc 17)."""
    try:
        ALGORITHMS.get(body.algorithm_id)
    except UnknownAlgorithmError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    session = getattr(request.app.state, "session", None)
    if session is not None:
        session.algorithm_id = body.algorithm_id
    return {"algorithm_id": body.algorithm_id}


def _load_env_from_payload(payload: SchedulerInputPayload) -> BackendSession:
    """Create env from validated payload, reset, return new session."""
    try:
        policy = ALGORITHMS.get(payload.algorithm_id or "rcw")
    except UnknownAlgorithmError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    env = SchedulerPlacementEnv(payload.to_env_dict())
    obs, info = env.reset(seed=42)
    session = BackendSession(
        env=env,
        algorithm_id=payload.algorithm_id or "rcw",
        reference_reward=None,
    )
    app.state.session = session
    return session


@app.post("/load")
def load(body: dict[str, Any]) -> dict[str, Any]:
    """Load scheduler input JSON and reset env. Body validated via SchedulerInputPayload."""
    try:
        payload = SchedulerInputPayload.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    if payload.algorithm_id is not None:
        try:
            ALGORITHMS.get(payload.algorithm_id)
        except UnknownAlgorithmError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
    session = _load_env_from_payload(payload)
    return _build_load_response(session)


@app.post("/load_example")
def load_example(body: LoadExampleRequest) -> dict[str, Any]:
    """Load an example by name (from fixtures)."""
    path = FIXTURES_DIR / f"{body.example}.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Example {body.example!r} not found. Examples: {_list_examples()}",
        )
    with path.open() as f:
        payload = SchedulerInputPayload.model_validate_json(f.read())
    if body.algorithm_id is not None:
        payload = payload.model_copy(update={"algorithm_id": body.algorithm_id})
    session = _load_env_from_payload(payload)
    session.algorithm_id = payload.algorithm_id or session.algorithm_id
    session.reference_reward = run_until_done(session.env, ALGORITHMS.get("rcw"))
    session.env.reset(seed=42)
    out = _build_load_response(session)
    out["reference_reward"] = session.reference_reward
    return out


@app.post("/step")
def step(body: StepRequest, request: Request) -> dict[str, Any]:
    """Execute one step. Returns observation, reward, terminated, info, full_state."""
    session = _get_session(request)
    obs, reward, terminated, truncated, info = session.env.step(body.action)
    return _build_step_response(session, obs, reward, terminated, truncated, info)


@app.post("/step_auto")
def step_auto(request: Request) -> dict[str, Any]:
    """Get action from current algorithm and execute one step (doc 17)."""
    session = _get_session(request)
    policy = ALGORITHMS.get(session.algorithm_id)
    action = policy(session.env)
    obs, reward, terminated, truncated, info = session.env.step(action)
    return _build_step_response(
        session, obs, reward, terminated, truncated, info, action=action
    )


@app.get("/state")
def state(request: Request) -> dict[str, Any]:
    """Return current full state for UI (includes algorithm_id, reference_reward if set)."""
    session = _get_session(request)
    return _build_state_response(session)
