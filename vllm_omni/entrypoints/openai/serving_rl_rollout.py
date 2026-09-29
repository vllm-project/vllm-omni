# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving layer for RL rollout (RFC #3747, P0).

P0 scope: world_model_env mode, single-session, DreamZero backbone.

P0 maps client rollout input onto DreamZero's DROID robot_obs schema and returns
DreamZero video latents explicitly. RGB/frame decoding can be added as a later
contract without pretending the latent is already an image observation.
"""

from __future__ import annotations

import binascii
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
    """Build DreamZero-compatible robot_obs."""
    robot_obs: dict[str, Any] = {"prompt": obs.prompt}

    if obs.images:
        robot_obs.update(obs.images)

    if obs.extra:
        for key, value in obs.extra.items():
            if key.startswith("observation/") or key in {"seed", "embodiment_name"}:
                robot_obs[key] = value

    joint_positions = action.joint_positions if action is not None else None
    gripper_position = action.gripper_position if action is not None else None

    if joint_positions is None and obs.state:
        joint_positions = obs.state[:7]
        if len(obs.state) > 7 and gripper_position is None:
            gripper_position = obs.state[7:8]

    if joint_positions is not None:
        joint_arr = np.asarray(joint_positions, dtype=np.float64).flatten()
        if joint_arr.size != 7:
            raise ValueError(f"DreamZero DROID joint_positions must have 7 values, got {joint_arr.size}.")
        robot_obs["observation/joint_position"] = joint_arr

    if gripper_position is not None:
        gripper_arr = np.asarray(gripper_position, dtype=np.float64).flatten()
        if gripper_arr.size != 1:
            raise ValueError(f"DreamZero DROID gripper_position must have 1 value, got {gripper_arr.size}.")
        robot_obs["observation/gripper_position"] = gripper_arr

    return robot_obs


def _encode_video_output(video: Any) -> dict[str, Any]:
    """Encode DreamZero video latent tensor / ndarray to JSON metadata."""
    if video is None:
        return {}
    if hasattr(video, "detach") and callable(video.detach):
        video = video.detach()
    if hasattr(video, "float") and callable(video.float):
        video = video.float()
    if hasattr(video, "cpu") and callable(video.cpu):
        video = video.cpu()
    if hasattr(video, "numpy"):
        video = video.numpy()
    if isinstance(video, np.ndarray):
        return {
            "video_latent": binascii.b2a_base64(video.tobytes(), newline=False).decode(),
            "shape": list(video.shape),
            "dtype": str(video.dtype),
            "encoding": "base64_raw_tensor",
        }
    return {"video_latent": str(video), "encoding": "string"}


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
        self._validate_model(req.model)
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
        # Next infer still passes reset=True; drop GPU KV now if the engine
        # session is reachable from this process.
        self._drop_engine_session(session_id)
        logger.info("Reset rollout session %s", session_id)
        return ResetSessionResponse(
            session_id=session_id,
            committed_step_id=session.committed_step_id,
        )

    async def close_session(self, session_id: str) -> None:
        await self._store.close(session_id)
        self._drop_engine_session(session_id)
        self._drop_engine_session(f"{session_id}:stateless")
        logger.info("Closed rollout session %s", session_id)

    async def get_status(self, session_id: str) -> SessionStatusResponse:
        session = await self._store.get(session_id, include_closed=True)
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
                    message = f"Step {req.step_id} is already committed; highest committed step is {committed}."
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
            engine_session_id = f"{session.session_id}:stateless"
        try:
            robot_obs = _merge_action_into_obs(req.observation, req.action)
            video_out = await self._infer_world_model(
                obs=robot_obs,
                session_id=engine_session_id,
                reset=reset,
            )
            next_observation = _encode_video_output(video_out)
        except ValueError as exc:
            logger.info("Invalid rollout step %d for session %s: %s", req.step_id, session.session_id, exc)
            return self._error_response(
                req.step_id,
                committed,
                session.context_length,
                "invalid_request",
                str(exc),
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
            next_observation=next_observation,
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
        identical to the policy_inference path. DreamZero's formatter peels
        ``video`` into ``result.images`` and leaves only ``actions`` on
        ``multimodal_output``; read images first and fall back to the raw key.
        """
        request = self._openpi.build_request(obs, session_id=session_id, reset=reset)
        result = None
        async for output in self._openpi.engine_client.generate(
            prompt=self._request_prompt(request),
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output

        if result is None:
            raise RuntimeError("World model request produced no output.")
        return self._extract_video_output(result)

    @staticmethod
    def _request_prompt(request: Any) -> Any:
        prompt = getattr(request, "prompt", None)
        if prompt is not None:
            return prompt
        return request.prompts[0]

    @staticmethod
    def _extract_video_output(result: Any) -> Any:
        """Prefer formatter-placed ``images``; fall back to multimodal video."""
        images = getattr(result, "images", None)
        if isinstance(images, (list, tuple)):
            if images:
                return images[0]
        elif images is not None and not isinstance(images, abc.Mapping):
            # Formatter assigns the peeled video tensor directly to images.
            return images

        multimodal_output = getattr(result, "multimodal_output", None)
        if isinstance(multimodal_output, abc.Mapping):
            video = multimodal_output.get("video")
            if video is not None:
                return video

        raise RuntimeError(
            "No video on result.images or multimodal_output['video']. "
            "Confirm DreamZero formatter placement for world_model_env."
        )

    def _drop_engine_session(self, session_id: str) -> None:
        drop = getattr(self._openpi, "drop_session", None)
        if not callable(drop):
            return
        try:
            drop(session_id)
        except Exception:
            logger.exception("Failed to drop engine session %s", session_id)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _validate_model(self, requested_model: str) -> None:
        served_model = getattr(self._openpi, "model_name", None)
        if served_model is None:
            return
        if requested_model == served_model:
            return
        if requested_model.lower() == "dreamzero" and "dreamzero" in served_model.lower():
            return
        raise ValueError(f"Requested rollout model {requested_model!r} does not match served model {served_model!r}.")

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
