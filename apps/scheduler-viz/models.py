# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for scheduler-viz API: request/response and scheduler input payload.

Validates incoming bodies and shapes responses; avoids dict[str, Any] through layers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ----- Scheduler input (load body) -----


class OpNodePayload(BaseModel):
    """Single node in op_graph JSON."""

    id: str
    op_type: str
    resource: str | None = None
    predecessors: list[str] = Field(default_factory=list)
    successors: list[str] = Field(default_factory=list)


class OpGraphPayload(BaseModel):
    """op_graph JSON shape."""

    nodes: list[OpNodePayload] = Field(default_factory=list)


class TopologyPayload(BaseModel):
    """topology JSON shape."""

    grid_cols: int = Field(..., ge=1)
    grid_rows: int = Field(..., ge=1)


class PlanPayload(BaseModel):
    """plan JSON shape (placement + grid)."""

    placement: dict[str, list[int]] = Field(default_factory=dict)
    grid_cols: int = Field(..., ge=1)
    grid_rows: int = Field(..., ge=1)


class SchedulerInputPayload(BaseModel):
    """Unified scheduler input for /load and /load_example. Validates JSON body."""

    op_graph: OpGraphPayload
    topology: TopologyPayload
    plan: PlanPayload
    algorithm_id: str | None = None

    def to_env_dict(self) -> dict[str, Any]:
        """Dict for SchedulerPlacementEnv (excludes algorithm_id)."""
        return self.model_dump(include={"op_graph", "topology", "plan"})


# ----- API requests -----


class LoadExampleRequest(BaseModel):
    """POST /load_example body."""

    example: str
    algorithm_id: str | None = None


class SetAlgorithmRequest(BaseModel):
    """POST /algorithm body."""

    algorithm_id: str


class StepRequest(BaseModel):
    """POST /step body."""

    action: int


# ----- API responses -----


class ObservationPayload(BaseModel):
    """observation in step/load response."""

    placement: list[list[int]]
    next_node: int


def observation_from_env(placement_ndarray: Any, next_node: int) -> ObservationPayload:
    """Build observation payload from env step/obs."""
    return ObservationPayload(
        placement=placement_ndarray.tolist(),
        next_node=int(next_node),
    )


class LoadResponse(BaseModel):
    """Response after /load or /load_example."""

    observation: ObservationPayload
    info: dict[str, Any]
    full_state: dict[str, Any]
    action_space_n: int
    algorithm_id: str
    reference_reward: float | None = None


class StepResponse(BaseModel):
    """Response after /step or /step_auto."""

    observation: ObservationPayload
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    full_state: dict[str, Any]
    action: int | None = None
    algorithm_id: str | None = None
    reference_reward: float | None = None


class StateResponse(BaseModel):
    """Response for /state."""

    full_state: dict[str, Any]
    algorithm_id: str
    reference_reward: float | None = None


# ----- Backend session (replaces globals) -----


class BackendSession(BaseModel):
    """Single session state: env, algorithm_id, reference_reward. Stored in app.state."""

    model_config = {"arbitrary_types_allowed": True}

    env: Any  # SchedulerPlacementEnv
    algorithm_id: str = "rcw"
    reference_reward: float | None = None
