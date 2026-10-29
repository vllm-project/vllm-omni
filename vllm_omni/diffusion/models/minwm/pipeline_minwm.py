# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""minWM Wan2.1 Action2V: camera-controlled causal DMD text-to-video.

The generator is the Self-Forcing backbone with a PRoPE camera branch in
every self-attention layer, four latent frames per AR block and a 20-frame
attention window. A request steers the camera with a minWM trajectory string
in ``extra_params["trajectory"]``; without one it renders without the camera
branch, as the reference does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.prope import PRoPECameras
from vllm_omni.diffusion.models.minwm.camera import trajectory_cameras
from vllm_omni.diffusion.models.self_forcing.pipeline_self_forcing import (
    SelfForcingPipeline,
    get_self_forcing_post_process_func,
)
from vllm_omni.errors import OmniClientError

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

__all__ = ["MinWMAction2VPipeline", "get_self_forcing_post_process_func"]

# 20 latent frames: the length the Action2V generator was distilled on.
MINWM_DEFAULT_NUM_FRAMES = 77


class MinWMAction2VPipeline(SelfForcingPipeline):
    """minWM Wan2.1 Action2V 4-step DMD generator with PRoPE camera control."""

    _default_num_frames: ClassVar[int] = MINWM_DEFAULT_NUM_FRAMES

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if not self.transformer.config.use_prope:
            raise ValueError("minWM Action2V needs a transformer checkpoint with use_prope=true.")

    def _parse_cameras(self, sampling: OmniDiffusionSamplingParams, num_latent_frames: int) -> PRoPECameras | None:
        extra_args = getattr(sampling, "extra_args", None) or {}
        if not isinstance(extra_args, Mapping):
            raise ValueError("sampling_params.extra_args must be a mapping.")
        trajectory = extra_args.get("trajectory")
        if trajectory is None:
            return None
        # Client errors, so the worker's rejection reaches the caller as a 400 rather than a 500.
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise OmniClientError("extra_params.trajectory must be a non-empty string such as 'a*4,w*8,s*7'.")
        try:
            viewmats, intrinsics = trajectory_cameras(trajectory, num_latent_frames=num_latent_frames)
        except ValueError as exc:
            raise OmniClientError(str(exc)) from exc
        # Built in float32: apply_prope casts to the feature dtype at the point of use, so only that last
        # multiply runs in bf16, not the composition and the rigid inverse.
        return PRoPECameras.from_views(viewmats.to(self.device), intrinsics.to(self.device), dtype=torch.float32)
