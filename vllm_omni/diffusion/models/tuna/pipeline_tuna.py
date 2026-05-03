# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tuna/Tuna-2 integration placeholder.

The upstream Tuna-2 project currently publishes research/inference code but no
released model weights or HuggingFace-style runtime package contract. vLLM-Omni
can still recognize Tuna configs and route them here so users get an actionable
message instead of a generic "unknown model" failure.
"""

from __future__ import annotations

from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

# TODO(#3303): Replace this placeholder once Tuna publishes a stable runtime
# loading contract that can be validated end to end in vLLM-Omni.
_TUNA_NOT_READY = (
    "Tuna/Tuna-2 is recognized by vLLM-Omni, but the runtime integration is "
    "not available yet. Track vLLM-Omni issue #3303 and the upstream "
    "facebookresearch/tuna-2 repository at "
    "https://github.com/facebookresearch/tuna-2. The upstream project "
    "currently uses its own Hydra-based inference entrypoint and checkpoint "
    "format, and does not publish full model weights or a stable "
    "HuggingFace/diffusers loading contract. To finish this integration, "
    "port Tuna2PixelPipeline/Tuna2RPixelPipeline/TunaPipeline into "
    "vllm_omni.diffusion.models.tuna and add checkpoint loading for the "
    "upstream .pt files."
)


def get_tuna_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(x):
        return x

    return post_process_func


class TunaExternalPipeline(nn.Module):
    """Recognized Tuna pipeline entrypoint.

    This class intentionally fails during initialization with a clear message.
    Keeping it registered lets model detection, stage-config resolution, and
    documentation converge before upstream releases a stable checkpoint/runtime
    contract that can be validated end to end.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        raise RuntimeError(_TUNA_NOT_READY)

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        raise RuntimeError(_TUNA_NOT_READY)

    def load_weights(self, *args: object, **kwargs: object) -> set[str]:
        raise RuntimeError(_TUNA_NOT_READY)
