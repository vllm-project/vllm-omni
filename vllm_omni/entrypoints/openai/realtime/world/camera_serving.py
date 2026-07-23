# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol structs for the /v1/realtime/world/* family of endpoints.

These are msgpack-serialised over the WebSocket wire via ``msgspec.msgpack``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_container(item) for item in value]
    return value


@dataclass(frozen=True)
class CameraServerConfig:
    """Static server-side camera/pipeline parameters sent to a client on connect."""

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> CameraServerConfig:
        return cls(_to_builtin_container(model_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)


class ServingRealtimeWorldCamera:
    """World Model Camera serving layer for OpenPI protocol.

    Model-specific transform/state lives in the diffusion pipeline.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self._current_session_id: str | None = None
        self._call_count = 0
        self.policy_server_config = self._get_policy_server_config(engine_client)
        self._force_reset = False

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeWorldCamera | None:
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("World Model OpenPI serving disabled for model %s", model_name)
            return None

    @staticmethod
    def _get_policy_server_config(engine_client: Any) -> CameraServerConfig:
        model_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                if getattr(stage_config, "stage_type", None) != "diffusion":
                    continue
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            od_config = getattr(engine_client, "od_config", None)
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            model_config = getattr(engine_client, "model_config", None)

        return CameraServerConfig.from_model_config(model_config)

    def reset(self, req: dict) -> None:
        """Reset serving state.

        Engine-side Lingbot state is reset on the next inference request via
        `extra_args["reset"]`, not by an immediate websocket-side RPC.
        """
        self._current_session_id = None
        self._force_reset = True

    async def infer(self, req: dict) -> np.ndarray:
        """raw req → engine → video."""
        # Session tracking

        session_id = req.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed %s → %s", self._current_session_id, session_id)
                self.reset({})
            self._current_session_id = session_id

        self._call_count += 1

        # Build request, run inference through AsyncOmni
        request = self._build_request(req)

        # After an inference call we reset the _force_reset argument
        self._force_reset = False

        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths.
        async for output in self.engine_client.generate(
            prompt=request.prompt,
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("World Model Camera OpenPI request produced no output.")

        return result

    def _build_request(self, req: dict) -> Any:
        """Build engine request from raw robot req.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        extra_args = {"session_id": self._current_session_id or "default", "force_reset": self._force_reset}

        camera = req.get("camera", None)

        multi_modal_data = {
            "image": req.get("image", None),
            "camera": camera,
        }

        prompt = req.get("prompt", "")

        extra_body = req.get("extra_body", {})

        height = extra_body.get("height", None)
        width = extra_body.get("width", None)
        num_frames = extra_body.get("num_frames", None)
        fps = extra_body.get("fps", None)
        seed = extra_body.get("seed", None)

        sampling_params = OmniDiffusionSamplingParams(
            height=height, width=width, num_frames=num_frames, frame_rate=fps, extra_args=extra_args, seed=seed
        )
        return OmniDiffusionRequest(
            prompt={
                "prompt": prompt,
                "multi_modal_data": multi_modal_data,
            },
            sampling_params=sampling_params,
            request_id=f"camera-{self._current_session_id or 'default'}",
        )
