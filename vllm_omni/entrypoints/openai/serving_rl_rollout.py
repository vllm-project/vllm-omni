# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving layer for RL rollout (RFC #3747, P0).

P0 scope: world_model_env mode, single-session, DreamZero backbone.

# Assumption: action -> obs folding
# DreamZero's diffusion pass jointly denoises (video_latent, action_latent)
# from random noise; it does NOT accept an action as a conditioning input at
# inference time. For world_model_env the client-provided Action is therefore
# treated as the previously-executed proprioceptive feedback and is
# concatenated with Observation.state before being passed as
# robot_obs["state"]. The model's video_out (predicted next frame) is returned
# as the next observation. This assumption MUST be validated against the merged
# DreamZero I/O schema (PR #2162) before implementing P1 batched rollouts.
"""

from __future__ import annotations

import base64
import time
from collections import abc
from typing import Any

import numpy as np
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.protocol.rollout import (
    Action,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorObject,
    Observation,
    ResetSessionResponse,
    RolloutStepRequest,
    RolloutStepResponse,
    SessionMetadata,
    SessionStatusResponse,
)
from vllm_omni.entrypoints.openai.rollout_session import (
    RolloutSession,
    RolloutSessionClosedError,
    RolloutSessionNotFoundError,
    RolloutSessionStore,
)
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI

logger = init_logger(__name__)


def _merge_action_into_obs(obs: Observation, action: Action | None) -> dict[str, Any]:
    """Build robot_obs dict, folding action into state (see module assumption)."""
    robot_obs: dict[str, Any] = {"prompt": obs.prompt}

    if obs.images:
        robot_obs.update(obs.images)

    if obs.extra:
        robot_obs.update(obs.extra)

    state_parts: list[list[float]] = []
    if obs.state:
        state_parts.append(obs.state)
    if action is not None and action.joint_positions:
        state_parts.append(action.joint_positions)
    if action is not None and action.extra:
        for v in action.extra.values():
            if isinstance(v, list):
                state_parts.append(v)

    if state_parts:
        robot_obs["state"] = np.concatenate([np.asarray(p, dtype=np.float64) for p in state_parts])

    return robot_obs


def _encode_video_output(video: Any) -> dict[str, Any]:
    """Encode raw video tensor / ndarray to a JSON-serialisable dict."""
    if video is None:
        return {}
    if hasattr(video, "detach") and callable(video.detach):
        video = video.detach()
    if hasattr(video, "cpu") and callable(video.cpu):
        video = video.cpu()
    if hasattr(video, "numpy"):
        video = video.numpy()
    if isinstance(video, np.ndarray):
        return {
            "video": base64.b64encode(video.tobytes()).decode(),
            "shape": list(video.shape),
            "dtype": str(video.dtype),
        }
    return {"video": str(video)}


class ServingRLRollout:
    """HTTP serving layer for RL rollout sessions.

    Wraps ServingRealtimeRobotOpenPI for world_model_env mode.
    Session state and committed_step_id tracking live in RolloutSessionStore.
    """

    def __init__(self, openpi_serving: ServingRealtimeRobotOpenPI) -> None:
        self._openpi = openpi_serving
        self._store = RolloutSessionStore()

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    async def create_session(self, req: CreateSessionRequest) -> CreateSessionResponse:
        session_id = random_uuid()
        session = await self._store.create(
            session_id=session_id,
            model=req.model,
            mode=req.mode,
        )
        logger.info("Created rollout session %s mode=%s", session_id, req.mode)
        return CreateSessionResponse(
            session_id=session_id,
            mode=session.mode,
            created_at=session.created_at,
        )

    async def reset_session(self, session_id: str) -> ResetSessionResponse:
        session = await self._store.reset(session_id)
        # Tell the engine to drop KV state for this session on the next call
        # by passing reset=True; no engine call is issued here.
        logger.info("Reset rollout session %s", session_id)
        return ResetSessionResponse(
            session_id=session_id,
            committed_step_id=session.committed_step_id,
        )

    async def close_session(self, session_id: str) -> None:
        await self._store.close(session_id)
        logger.info("Closed rollout session %s", session_id)

    async def get_status(self, session_id: str) -> SessionStatusResponse:
        session = await self._store.get(session_id)
        return SessionStatusResponse(
            session_id=session_id,
            committed_step_id=session.committed_step_id,
            context_length=session.context_length,
            closed=session.closed,
        )

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #

    async def step(
        self,
        session_id: str,
        req: RolloutStepRequest,
    ) -> RolloutStepResponse:
        try:
            session = await self._store.get(session_id)
        except RolloutSessionNotFoundError:
            return self._error_response(
                req.step_id,
                -1,
                0,
                "session_not_found",
                f"Session {session_id!r} does not exist.",
            )
        except RolloutSessionClosedError:
            return self._error_response(
                req.step_id,
                -1,
                0,
                "session_closed",
                f"Session {session_id!r} is closed.",
            )

        async with session.lock:
            return await self._run_step(session, req)

    async def _run_step(
        self,
        session: RolloutSession,
        req: RolloutStepRequest,
    ) -> RolloutStepResponse:
        t0 = time.perf_counter()
        committed = session.committed_step_id

        if req.use_session_context:
            expected_step_id = committed + 1
            if req.step_id != expected_step_id:
                if req.step_id <= committed:
                    code = "step_already_committed"
                    message = (
                        f"Step {req.step_id} is already committed; "
                        f"highest committed step is {committed}."
                    )
                else:
                    code = "step_out_of_order"
                    message = f"Expected step_id {expected_step_id}, got {req.step_id}."
                return self._error_response(
                    req.step_id,
                    committed,
                    session.context_length,
                    code,
                    message,
                )

        # reset=True on first call after session create/reset
        reset = committed == -1 or not req.use_session_context
        engine_session_id = session.session_id
        if not req.use_session_context:
            engine_session_id = f"{session.session_id}:stateless:{random_uuid()}"
        robot_obs = _merge_action_into_obs(req.observation, req.action)

        try:
            video_out = await self._infer_world_model(
                obs=robot_obs,
                session_id=engine_session_id,
                reset=reset,
            )
        except Exception as exc:
            logger.exception("Step %d failed for session %s", req.step_id, session.session_id)
            # Do NOT advance committed_step_id on failure (RFC section 6.5).
            return self._error_response(
                req.step_id,
                committed,
                session.context_length,
                "inference_error",
                str(exc),
            )

        if req.use_session_context:
            # Only advance committed_step_id on successful contextual steps.
            await self._store.advance(session.session_id, req.step_id)
            committed = req.step_id

        latency_ms = (time.perf_counter() - t0) * 1000.0
        metadata = SessionMetadata(
            latency_ms=round(latency_ms, 2),
            steps_generated=1,
            context_length=session.context_length,
            committed_step_id=committed,
        )
        return RolloutStepResponse(
            step_id=req.step_id,
            next_observation=_encode_video_output(video_out),
            model_metadata=metadata,
        )

    async def _infer_world_model(
        self,
        obs: dict[str, Any],
        *,
        session_id: str,
        reset: bool,
    ) -> Any:
        """Call the engine and return the video (next-observation) output.

        Reuses ServingRealtimeRobotOpenPI.build_request() so request routing,
        session_id threading, and OmniDiffusionSamplingParams construction are
        identical to the policy_inference path. Only the output extraction
        differs: we pull multimodal_output["video"] instead of ["actions"].
        """
        request = self._openpi.build_request(obs, session_id=session_id, reset=reset)
        result = None
        async for output in self._openpi.engine_client.generate(
            prompt=request.prompts[0],
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output

        if result is None:
            raise RuntimeError("World model request produced no output.")

        multimodal_output = getattr(result, "multimodal_output", None)
        if not isinstance(multimodal_output, abc.Mapping):
            raise RuntimeError("Missing multimodal_output in world model result.")

        video = multimodal_output.get("video")
        if video is None:
            raise RuntimeError(
                "multimodal_output['video'] is None. "
                "Confirm DreamZero returns video in world_model_env mode (see module assumption)."
            )
        return video

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _error_response(
        step_id: int,
        committed_step_id: int,
        context_length: int,
        code: str,
        message: str,
    ) -> RolloutStepResponse:
        metadata = SessionMetadata(
            latency_ms=0.0,
            steps_generated=0,
            context_length=context_length,
            committed_step_id=committed_step_id,
        )
        return RolloutStepResponse(
            step_id=step_id,
            next_observation=None,
            model_metadata=metadata,
            error=ErrorObject(
                code=code,
                message=message,
                step_id=step_id,
                committed_step_id=committed_step_id,
            ),
        )
