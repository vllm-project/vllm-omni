# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → transform (dataset key mapping) → unified obs →
`DiffusionEngine.step()` → actions.
Transform is stateless and selected per-request via `obs["embodiment"]`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.realtime.robot.transform.base import (
    RobotPolicyTransform,
    get_transform,
)

logger = init_logger(__name__)

# Default embodiment when not specified in obs
DEFAULT_EMBODIMENT = "roboarena"


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer for OpenPI protocol.

    Stateless transform routes by obs["embodiment"].
    Model-specific state (frame buffer, KV cache) lives in pipeline.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
        default_embodiment: str = DEFAULT_EMBODIMENT,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self.default_embodiment = default_embodiment
        self._current_session_id: str | None = None
        self._call_count = 0

        # Ensure default transforms are registered
        self._ensure_transforms_loaded()

    @staticmethod
    def _ensure_transforms_loaded() -> None:
        """Import transform modules to trigger register_transform calls."""
        import vllm_omni.entrypoints.openai.realtime.robot.transform.droid  # noqa: F401
        import vllm_omni.entrypoints.openai.realtime.robot.transform.roboarena  # noqa: F401

    def reset(self, obs: dict) -> None:
        """Reset session state."""
        self._call_count = 0
        self._current_session_id = None

    async def infer(self, obs: dict) -> np.ndarray:
        """raw obs → transform → engine → actions."""
        # Session tracking
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed %s → %s", self._current_session_id, session_id)
                self.reset({})
            self._current_session_id = session_id

        self._call_count += 1

        # Transform: dataset format → unified format
        transform = self._get_transform(obs)
        unified_obs = transform.transform_input(obs)

        # Build request, run inference through AsyncOmni
        request = self._build_request(unified_obs)
        result = None
        async for output in self.engine_client.generate(
            prompt=request.prompts[0],
            request_id=request.request_ids[0],
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        # Extract actions (via transform or default)
        return self._extract_actions(result, transform)

    def _get_transform(self, obs: dict) -> RobotPolicyTransform:
        """Select transform by obs['embodiment'] or default."""
        embodiment = obs.get("embodiment", self.default_embodiment)
        return get_transform(embodiment)

    def _build_request(self, unified_obs: dict) -> Any:
        """Build engine request from unified obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        extra_args = {
            "reset": self._call_count <= 1,
            "session_id": self._current_session_id or "default",
            "unified_obs": unified_obs,
        }

        prompt = unified_obs["prompt"]
        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[f"robot-{self._current_session_id or 'default'}"],
        )

    def _extract_actions(self, result: Any, transform: RobotPolicyTransform) -> np.ndarray:
        """Extract actions from engine result."""
        if hasattr(result, "__iter__"):
            result = list(result)
            if result:
                result = result[0]

        actions = transform.transform_output(result)
        if isinstance(actions, torch.Tensor):
            return actions.cpu().float().numpy()
        return np.asarray(actions, dtype=np.float32)
