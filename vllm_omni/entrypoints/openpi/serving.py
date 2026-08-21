# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

from vllm_omni.outputs.output_metadata import (
    DiffusionMetadataMapping,
    validate_public_diffusion_metadata,
)

logger = init_logger(__name__)

ActionOutput = np.ndarray | dict[str, np.ndarray]


@dataclass(frozen=True)
class RobotPolicyActionOutput:
    """Validated OpenPI robot-policy action output.

    OpenPI currently returns only the action payload on the websocket. The
    metadata is still validated here so serving tests and future endpoint
    adapters can use the same contract.
    """

    actions: ActionOutput
    action_metadata: DiffusionMetadataMapping


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_container(item) for item in value]
    return value


def validate_robot_policy_action_output(multimodal_output: Any) -> RobotPolicyActionOutput:
    """Validate and normalize a robot-policy multimodal output.

    The serving contract expects formatter-normalized actions at
    ``multimodal_output["actions"]`` and optional public action metadata at
    ``multimodal_output["metadata"]["actions"]``.
    """
    if not isinstance(multimodal_output, Mapping):
        raise RuntimeError("Missing multimodal_output in robot policy result")

    if "actions" not in multimodal_output or multimodal_output.get("actions") is None:
        raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")

    actions = _normalize_action_payload(multimodal_output["actions"])
    action_metadata = _extract_action_metadata(multimodal_output)
    _validate_action_metadata_matches_payload(actions, action_metadata)
    return RobotPolicyActionOutput(actions=actions, action_metadata=action_metadata)


def _normalize_action_payload(actions: Any) -> ActionOutput:
    if isinstance(actions, Mapping):
        if not actions:
            raise ValueError("Robot policy action payload mapping must be non-empty.")
        return {
            str(key): _as_float32_action_array(value, f"multimodal_output['actions'][{key!r}]")
            for key, value in actions.items()
        }
    return _as_float32_action_array(actions, "multimodal_output['actions']")


def _as_float32_action_array(value: Any, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric and convertible to float32.") from exc
    if array.ndim == 0:
        raise ValueError(f"{field} must be an action array, not a scalar.")
    if array.size == 0:
        raise ValueError(f"{field} must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values.")
    return array


def _extract_action_metadata(multimodal_output: Mapping[str, Any]) -> DiffusionMetadataMapping:
    metadata = multimodal_output.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("multimodal_output['metadata'] must be a mapping when present.")
    validate_public_diffusion_metadata(metadata)

    action_metadata = metadata.get("actions", {})
    if action_metadata is None:
        return {}
    if not isinstance(action_metadata, Mapping):
        raise TypeError("multimodal_output['metadata']['actions'] must be a mapping when present.")
    return dict(action_metadata)


def _validate_action_metadata_matches_payload(
    actions: ActionOutput,
    action_metadata: DiffusionMetadataMapping,
) -> None:
    expected_horizon = _optional_int_metadata(action_metadata, "horizon")
    if expected_horizon is None:
        expected_horizon = _optional_int_metadata(action_metadata, "action_horizon")
    expected_dim = _optional_int_metadata(action_metadata, "action_dim")
    valid_steps = _optional_int_metadata(action_metadata, "valid_steps")

    has_named_actions = isinstance(actions, Mapping)
    arrays = actions.values() if has_named_actions else (actions,)
    for array in arrays:
        horizon = _action_horizon(array)
        action_dim = _action_dim(array)
        if expected_horizon is not None and horizon is not None and horizon != expected_horizon:
            raise ValueError(
                "Robot policy action metadata horizon does not match action payload: "
                f"metadata={expected_horizon}, payload={horizon}."
            )
        if not has_named_actions and expected_dim is not None and action_dim is not None and action_dim != expected_dim:
            raise ValueError(
                "Robot policy action metadata action_dim does not match action payload: "
                f"metadata={expected_dim}, payload={action_dim}."
            )
        if valid_steps is not None and horizon is not None and valid_steps > horizon:
            raise ValueError(
                "Robot policy action metadata valid_steps exceeds action horizon: "
                f"valid_steps={valid_steps}, horizon={horizon}."
            )


def _optional_int_metadata(metadata: DiffusionMetadataMapping, key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    return int(value)


def _action_horizon(action: np.ndarray) -> int | None:
    if action.ndim >= 2:
        return int(action.shape[-2])
    if action.ndim == 1:
        return 1
    return None


def _action_dim(action: np.ndarray) -> int | None:
    if action.ndim >= 1:
        return int(action.shape[-1])
    return None


@dataclass(frozen=True)
class PolicyServerConfig:
    """OpenPI policy server handshake config.

    Values are model-specific and must be provided by the loaded policy model.
    """

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> PolicyServerConfig:
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)

        if raw_config is None:
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        if isinstance(raw_config, cls):
            return raw_config
        if not isinstance(raw_config, Mapping):
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
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
        self.policy_server_config = self._get_policy_server_config(engine_client)
        self._request_counter = count()

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeRobotOpenPI | None:
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
        """Compatibility hook; per-connection state lives in RobotRealtimeConnection."""

    async def infer(self, obs: dict, *, session_id: str, reset: bool) -> ActionOutput:
        """raw obs → engine → actions."""
        # Build request, run inference through AsyncOmni
        request = self._build_request(obs, session_id=session_id, reset=reset)
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
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_actions(result)

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    def _build_request(self, obs: dict, *, session_id: str, reset: bool) -> Any:
        """Build engine request from raw robot obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": obs,
        }

        prompt = obs.get("prompt", "")
        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        return OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=self._next_request_id(session_id),
        )

    def _extract_actions(self, result: Any) -> ActionOutput:
        """Extract actions from engine result."""
        multimodal_output = getattr(result, "multimodal_output", None)
        return validate_robot_policy_action_output(multimodal_output).actions
