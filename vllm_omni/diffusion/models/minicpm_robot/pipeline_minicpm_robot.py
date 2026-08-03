# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-RobotManip single-stage diffusion pipeline for vLLM-Omni."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.minicpm_robot.policy import (
    MiniCPMRobotPolicy,
    normalize_robot_obs,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)


def _to_float32_action_dict(
    actions: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    converted = {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
    if not converted:
        raise RuntimeError("MiniCPM-RobotManip policy returned an empty action dict.")
    return converted


class MiniCPMRobotManipPipeline(nn.Module):
    """Thin wrapper around MiniCPM-RobotManip VLA policy.

    Observations arrive through
    ``sampling_params.extra_args["robot_obs"]``, this pipeline delegates
    to ``MiniCPMRobotPolicy.get_action()`` and returns actions via
    ``DiffusionOutput.output["actions"]``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        model_config = od_config.model_config
        self.model_path = od_config.model
        self.embodiment_id = int(model_config.get("embodiment_id", 0))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        processor_path = model_config.get("processor_path")
        prompt_template = model_config.get("prompt_template")

        logger.info(
            "Loading MiniCPM-RobotManip policy from %s with embodiment_id=%d",
            self.model_path,
            self.embodiment_id,
        )
        self.policy = MiniCPMRobotPolicy(
            model_path=self.model_path,
            device=self.device,
            processor_path=processor_path,
            prompt_template=prompt_template,
            embodiment_id=self.embodiment_id,
        )
        self._validate_policy_server_config(model_config.get("policy_server_config"))

    def _validate_policy_server_config(self, psc: Mapping[str, Any] | None) -> None:
        """Fail fast if the deploy YAML handshake drifts from the loaded model.

        ``policy_server_config`` is sent verbatim to the OpenPI client, so its
        model-specific values must match the loaded checkpoint's config.
        """
        if not isinstance(psc, Mapping):
            return
        model_cfg = self.policy.model.config
        expected = {
            "action_horizon": model_cfg.action_horizon,
            "action_dim": model_cfg.action_dim,
            "state_dim": model_cfg.state_dim,
        }
        for key, expected_val in expected.items():
            if key in psc and psc[key] != expected_val:
                raise ValueError(f"policy_server_config.{key}={psc[key]} != loaded model's {key}={expected_val}.")

    def reset(self) -> dict[str, Any]:
        return self.policy.reset() or {}

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                "MiniCPMRobotManipPipeline.load_weights received "
                f"{len(consumed)} weight tensors; "
                "weights_sources=() should prevent this. "
                "Weights are loaded directly by MiniCPMRobotPolicy "
                "via AutoModel.from_pretrained()."
            )
        return set()

    def _dummy_actions(self) -> dict[str, np.ndarray]:
        action_dim = self.policy.model.config.action_dim
        action_horizon = self.policy.model.config.action_horizon
        return {
            "default": np.zeros((1, action_horizon, action_dim), dtype=np.float32),
        }

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        del kwargs
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            if req.request_id == DUMMY_DIFFUSION_REQUEST_ID:
                return DiffusionOutput(output={"actions": self._dummy_actions()})
            return DiffusionOutput(
                error=("MiniCPMRobotManipPipeline.forward expects sampling_params.extra_args['robot_obs'].")
            )
        if not isinstance(robot_obs, Mapping):
            return DiffusionOutput(error=(f"robot_obs must be a dict, got {type(robot_obs).__name__}."))

        if extra_args.get("reset"):
            self.reset()

        state_dim = int(getattr(self.policy, "state_dim", 80))
        try:
            normalized_obs = normalize_robot_obs(robot_obs, state_dim=state_dim)
        except (TypeError, ValueError) as exc:
            return DiffusionOutput(error=str(exc))

        seed = getattr(req.sampling_params, "seed", None)
        actions = self.policy.get_action(normalized_obs, seed=seed)
        if not isinstance(actions, Mapping):
            return DiffusionOutput(
                error=(f"MiniCPM-RobotManip policy returned {type(actions).__name__}; expected dict actions.")
            )
        return DiffusionOutput(output={"actions": _to_float32_action_dict(actions)})
