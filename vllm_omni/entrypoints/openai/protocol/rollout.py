# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pydantic schemas for RL rollout serving (RFC #3747).

P0 scope: world_model_env mode only (observation + action -> next observation).

P0 maps rollout observations/actions onto DreamZero's DROID robot_obs schema:
camera images retain their DreamZero observation/* keys, joint/gripper state is
passed through observation/joint_position and observation/gripper_position, and
the returned video payload is the DreamZero VAE latent unless a later endpoint
adds explicit latent decoding.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Single-step robot observation."""

    images: dict[str, list] | None = Field(
        default=None,
        description="Named DreamZero camera images as nested numeric lists. Base64 strings are not accepted in P0.",
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
    """Executed action for world_model_env conditioning."""

    joint_positions: list[float] | None = Field(
        default=None,
        description="Executed joint positions; forwarded as observation/joint_position for DreamZero.",
    )
    gripper_position: float | list[float] | None = Field(
        default=None,
        description="Executed gripper position; forwarded as observation/gripper_position for DreamZero.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    latency_ms: float
    steps_generated: int
    context_length: int
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
        description="Predicted next observation metadata. P0 returns DreamZero video_latent, not RGB frames.",
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
