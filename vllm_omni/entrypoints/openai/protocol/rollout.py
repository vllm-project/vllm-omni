# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic schemas for RL rollout serving (RFC #3747).

P0 scope: world_model_env mode only (observation + action -> next observation).

Assumption: the client-provided Action is NOT fed as a denoising-time input to
DreamZero. Instead it is concatenated with the Observation.state vector and
passed as robot_obs["state"] (proprioception) on the NEXT step. DreamZero's
diffusion pass jointly generates (next_video, predicted_action); world_model_env
uses the next_video output and ignores predicted_action. This assumption must be
validated against the merged DreamZero I/O schema before P1.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Single-step robot observation."""

    images: dict[str, list | str] | None = Field(
        default=None,
        description="Named camera images as nested float lists (H, W, C) or base64 strings.",
    )
    state: list[float] | None = Field(
        default=None,
        description="Proprioceptive state vector (joint positions, velocities, etc.).",
    )
    prompt: str = Field(default="", description="Optional language conditioning.")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Pass-through fields forwarded verbatim to robot_obs.",
    )


class Action(BaseModel):
    """Executed action for world_model_env conditioning (see module assumption)."""

    joint_positions: list[float] | None = Field(
        default=None,
        description="Executed joint positions; concatenated with Observation.state.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    latency_ms: float
    steps_generated: int
    session_context_length: int
    committed_step_id: int = Field(
        description="Highest step_id whose context has been atomically committed. "
        "-1 means no step has been committed yet (fresh or reset session).",
    )
    session_memory_bytes: int | None = None
    uncertainty: float | None = None


class ErrorObject(BaseModel):
    code: str
    message: str
    step_id: int | None = None
    committed_step_id: int = -1


class CreateSessionRequest(BaseModel):
    model: str
    mode: Literal["world_model_env"] = "world_model_env"


class CreateSessionResponse(BaseModel):
    session_id: str
    mode: str
    created_at: float


class RolloutStepRequest(BaseModel):
    step_id: int = Field(ge=0, description="Monotonically increasing per session.")
    observation: Observation
    action: Action = Field(
        description="Required for world_model_env: executed action that produced this observation.",
    )
    use_session_context: bool = True


class RolloutStepResponse(BaseModel):
    step_id: int
    next_observation: dict[str, Any] | None = Field(
        default=None,
        description="Predicted next observation. Contains 'video' key with base64-encoded frames.",
    )
    model_metadata: SessionMetadata
    error: ErrorObject | None = None


class ResetSessionResponse(BaseModel):
    session_id: str
    committed_step_id: int


class SessionStatusResponse(BaseModel):
    session_id: str
    committed_step_id: int
    context_length: int
    closed: bool
