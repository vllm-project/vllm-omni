# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen's host-side latent geometry shared by preparation and admission."""

from dataclasses import dataclass

from vllm_omni.diffusion.mixfusion.scheduling import MixFusionRequestInfo
from vllm_omni.diffusion.request import OmniDiffusionRequest


@dataclass(frozen=True)
class QwenMixFusionConfig:
    vae_scale_factor: int
    default_sample_size: int
    channels: int


def image_token_grid(height: int, width: int, vae_scale_factor: int) -> tuple[int, int]:
    return int(height) // (vae_scale_factor * 2), int(width) // (vae_scale_factor * 2)


def prepare_mixfusion_request(
    request: OmniDiffusionRequest,
    config: QwenMixFusionConfig,
    min_chunk_tokens: int,
    max_chunks: int,
) -> MixFusionRequestInfo | None:
    sampling = request.sampling_params
    prompt = request.prompt
    # These inputs need a separate model preparation contract before they can
    # participate in heterogeneous packing. Plain text dictionaries are fine.
    if not isinstance(prompt, str):
        if not isinstance(prompt, dict) or set(prompt) - {"prompt", "negative_prompt"}:
            return None
    if sampling.latents is not None:
        return None
    height = sampling.height or config.default_sample_size * config.vae_scale_factor
    width = sampling.width or config.default_sample_size * config.vae_scale_factor
    h, w = image_token_grid(height, width, config.vae_scale_factor)
    num_images = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
    negative = isinstance(prompt, dict) and prompt.get("negative_prompt") is not None
    true_cfg = (sampling.true_cfg_scale or 4.0) > 1 and negative
    return MixFusionRequestInfo(
        (num_images, h * w, config.channels), min_chunk_tokens, max_chunks, condition_key=(true_cfg,)
    )
