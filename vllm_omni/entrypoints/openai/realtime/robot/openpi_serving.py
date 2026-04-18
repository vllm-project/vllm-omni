# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → `DiffusionEngine.step()` → actions.
DreamZero owns dataset transforms inside the diffusion pipeline.
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
class PolicyServerConfig:
    """OpenPI policy server handshake config.

    Values are model-specific and must be provided by the loaded policy model.
    """

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> "PolicyServerConfig":
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)

        if raw_config is None:
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        if isinstance(raw_config, cls):
            return raw_config
        if not isinstance(raw_config, Mapping):
            raise TypeError("policy_server_config must be a dict.")
        return cls(_to_builtin_container(raw_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer for OpenPI protocol.

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

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> "ServingRealtimeRobotOpenPI | None":
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("Robot OpenPI serving disabled for model %s", model_name)
            return None

    @staticmethod
    def _get_policy_server_config(engine_client: Any) -> PolicyServerConfig:
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
        return PolicyServerConfig.from_model_config(model_config)

    def reset(self, obs: dict) -> None:
        """Reset serving state.

        Engine-side DreamZero state is reset on the next inference request via
        `extra_args["reset"]`, not by an immediate websocket-side RPC.
        """
        self._call_count = 0
        self._current_session_id = None

    async def infer(self, obs: dict) -> np.ndarray:
        """raw obs → engine → actions."""
        # Session tracking
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed %s → %s", self._current_session_id, session_id)
                self.reset({})
            self._current_session_id = session_id

        self._call_count += 1

        # Build request, run inference through AsyncOmni
        request = self._build_request(obs)
        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths.
        async for output in self.engine_client.generate(
            prompt=request.prompts[0],
            request_id=request.request_ids[0],
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_actions(result)

    def _build_request(self, obs: dict) -> Any:
        """Build engine request from raw robot obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        # `_call_count` is reset by websocket reset/session switches, then
        # incremented before this request is built. DreamZero pipeline consumes
        # this flag and clears its frame buffer / KV cache before accumulation.
        extra_args = {
            "reset": self._call_count <= 1,
            "session_id": self._current_session_id or "default",
            "robot_obs": obs,
        }

        prompt = obs.get("prompt", "")
        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[f"robot-{self._current_session_id or 'default'}"],
        )

    def _extract_actions(self, result: Any) -> np.ndarray:
        """Extract actions from engine result."""
        if hasattr(result, "__iter__"):
            result = list(result)
            if result:
                result = result[0]

        if not hasattr(result, "multimodal_output") or result.multimodal_output is None:
            raise RuntimeError("Missing multimodal_output in robot policy result")

        actions = result.multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        return np.asarray(actions, dtype=np.float32)
