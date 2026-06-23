# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest

logger = init_logger(__name__)


def _to_float32_action_dict(actions: Mapping[str, Any]) -> dict[str, np.ndarray]:
    converted = {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
    if not converted:
        raise RuntimeError("GR00T policy returned an empty action dict.")
    return converted


class Gr00tN1d7Pipeline(nn.Module):
    """GR00T N1.7 policy pipeline backed by vLLM-Omni's local GR00T port.

    vLLM-Omni owns the serving integration: OpenPI observations arrive through
    `sampling_params.extra_args["robot_obs"]`, this pipeline runs GR00T policy
    inference, and actions are returned through `DiffusionOutput.output["actions"]`.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = od_config.model_config
        self.model_path = od_config.model
        self.embodiment_tag = str(model_config.get("embodiment_tag") or "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
        self.strict = bool(model_config.get("strict", True))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading GR00T N1.7 policy from %s with embodiment_tag=%s", self.model_path, self.embodiment_tag)
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag=self.embodiment_tag,
            device=self.device,
            strict=self.strict,
        )
        self._validate_policy_server_config(model_config.get("policy_server_config"))

    def _validate_policy_server_config(self, psc: Mapping[str, Any] | None) -> None:
        """Fail fast if the deploy handshake drifts from the loaded checkpoint.

        ``policy_server_config`` is sent verbatim to the OpenPI client, so its
        model/embodiment-specific values must match what the loaded policy actually
        produces; otherwise the client is handed the wrong action contract.
        """
        if not isinstance(psc, Mapping):
            return
        action_config = self.policy.modality_configs["action"]
        expected_horizon = len(action_config.delta_indices)
        expected_keys = set(action_config.modality_keys)

        if "action_horizon" in psc and psc["action_horizon"] != expected_horizon:
            raise ValueError(
                f"policy_server_config.action_horizon={psc['action_horizon']} != loaded model's "
                f"action horizon {expected_horizon}."
            )
        if "action_keys" in psc and set(psc["action_keys"]) != expected_keys:
            raise ValueError(
                f"policy_server_config.action_keys={list(psc['action_keys'])} != loaded model's "
                f"action keys {sorted(expected_keys)}."
            )
        psc_embodiment = psc.get("embodiment_tag")
        if psc_embodiment is not None and psc_embodiment != self.embodiment_tag:
            raise ValueError(
                f"policy_server_config.embodiment_tag={psc_embodiment!r} != "
                f"model_config.embodiment_tag={self.embodiment_tag!r}."
            )

    def reset(self) -> dict[str, Any]:
        return self.policy.reset() or {}

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"Gr00tN1d7Pipeline.load_weights received {len(consumed)} weight tensors; "
                "weights_sources=() should prevent this. GR00T weights are loaded directly by Gr00tPolicy."
            )
        return set()

    def _dummy_actions(self) -> dict[str, np.ndarray]:
        embodiment_value = self.policy.embodiment_tag.value
        action_config = self.policy.modality_configs["action"]
        horizon = len(action_config.delta_indices)
        norm_params = self.policy.processor.state_action_processor.norm_params[embodiment_value]["action"]
        actions = {}
        for key in action_config.modality_keys:
            dim = norm_params[key]["dim"]
            dim = int(dim.item() if hasattr(dim, "item") else dim)
            actions[key] = np.zeros((1, horizon, dim), dtype=np.float32)
        return actions

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        del kwargs
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            if req.request_id == DUMMY_DIFFUSION_REQUEST_ID:
                return DiffusionOutput(output={"actions": self._dummy_actions()})
            return DiffusionOutput(error="Gr00tN1d7Pipeline.forward expects sampling_params.extra_args['robot_obs'].")
        if not isinstance(robot_obs, Mapping):
            return DiffusionOutput(error=f"robot_obs must be a dict, got {type(robot_obs).__name__}.")

        if extra_args.get("reset"):
            self.reset()

        policy_obs = _normalize_observation(robot_obs, language_key=self.policy.language_key)
        result = self.policy.get_action(policy_obs)
        actions = result[0] if isinstance(result, tuple) else result
        if not isinstance(actions, Mapping):
            return DiffusionOutput(error=f"GR00T policy returned {type(actions).__name__}; expected dict actions.")
        # Return actions via output.output (like the DreamZero OpenPI policy) so the engine's
        # empty-output guard passes.
        return DiffusionOutput(output={"actions": _to_float32_action_dict(actions)})


def _normalize_observation(robot_obs: Mapping[str, Any], *, language_key: str) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    if "video" in robot_obs:
        obs["video"] = robot_obs["video"]
    elif "images" in robot_obs:
        obs["video"] = robot_obs["images"]
    if "state" in robot_obs:
        obs["state"] = robot_obs["state"]
    if "language" in robot_obs:
        obs["language"] = robot_obs["language"]
    else:
        prompt = robot_obs.get("prompt")
        if prompt is not None:
            obs["language"] = {language_key: [[str(prompt)]]}
    return obs
