# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 (medium) text-to-audio pipeline for vLLM-Omni.

Issue: https://github.com/vllm-project/vllm-omni/issues/3787
Reference impl: https://github.com/Stability-AI/stable-audio-3 (MIT)

This is a Path B (custom/no-diffusers) integration. The structure mirrors
`vllm_omni/diffusion/models/stable_audio/pipeline_stable_audio.py` (Stable
Audio Open 1.0), with three deltas:

  1. SAME autoencoder (in same_autoencoder.py) replaces AutoencoderOobleck.
  2. Variable-length latents sized to the requested duration.
  3. DiT ported in stable_audio_3_transformer.py — no diffusers fallback.

SCOPE (v1): text-to-audio only. audio-to-audio editing and inpainting are
deferred.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.stable_audio_3.same_autoencoder import SAMEAutoencoder
from vllm_omni.diffusion.models.stable_audio_3.stable_audio_3_transformer import (
    StableAudio3DiTModel,
    StableAudio3SchedulerWrapper,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

logger = init_logger(__name__)


def get_stable_audio_3_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-process raw audio tensor → numpy array (factory for the registry)."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type in ("latent", "pt"):
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


# ---------------------------------------------------------------------------
# USER DECISION #3 — Text encoder choice
# ---------------------------------------------------------------------------
# Stable Audio Open 1.0 used T5 (T5EncoderModel + T5TokenizerFast) plus a
# StableAudioProjectionModel for duration conditioning.
#
# SA3 may use the same stack or something different. Check the upstream
# `model_index.json` or `text_encoder/` subfolder once you have the weights.
#
# Likely options:
#   A. T5-base / T5-large       — same as SA Open 1.0, transformers handles loading
#   B. CLAP / MuLan style         — joint audio-text, requires a different encoder
#   C. Custom embedding-only      — upstream ships a projection-only model
#
# Set _TEXT_ENCODER_KIND below once known. The pipeline branches on this.
# ---------------------------------------------------------------------------
_TEXT_ENCODER_KIND: str = "t5"  # TODO(stable-audio-3): confirm against upstream model_index.json


class StableAudio3Pipeline(nn.Module, SupportAudioOutput, DiffusionPipelineProfilerMixin):
    """Stable Audio 3 text-to-audio pipeline.

    Engine contract:
      - Inherits SupportAudioOutput  → engine routes outputs as audio waveforms
      - Inherits DiffusionPipelineProfilerMixin → optional profiling
      - Implements forward(req: OmniDiffusionRequest) → DiffusionOutput
      - Defines self.weights_sources → DiffusersPipelineLoader knows where to load from
      - Implements load_weights() → AutoWeightsLoader entry point
    """

    # Picked up by `supports_audio_output` in the diffusion engine.
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 44100

    # SA3 Medium per issue #3787:
    #   - up to 380s clips (vs SA Open 1.0's ~47s)
    #   - peak ~6.5 GB VRAM (Medium)
    #   - requires Flash Attention 2
    max_audio_seconds: ClassVar[float] = 380.0

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.float16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # ------------------------------------------------------------------
        # Weight loader hook — DiT weights live under `transformer/` subfolder.
        # If upstream packages SA3 differently (single safetensors at root),
        # switch to BAGEL's pattern: subfolder=None, prefix="" — and remove
        # `fall_back_to_pt`.
        # ------------------------------------------------------------------
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # ------------------------------------------------------------------
        # Text encoder + tokenizer
        # ------------------------------------------------------------------
        # TODO(stable-audio-3): implement encoder load based on _TEXT_ENCODER_KIND.
        # See USER DECISION #3 above.
        if _TEXT_ENCODER_KIND == "t5":
            from transformers import T5EncoderModel, T5TokenizerFast

            self.tokenizer = T5TokenizerFast.from_pretrained(
                model, subfolder="tokenizer", local_files_only=local_files_only,
            )
            self.text_encoder = T5EncoderModel.from_pretrained(
                model, subfolder="text_encoder",
                torch_dtype=dtype, local_files_only=local_files_only,
            ).to(self.device)
        else:
            raise NotImplementedError(
                f"_TEXT_ENCODER_KIND={_TEXT_ENCODER_KIND!r} not yet wired. "
                "Edit pipeline_stable_audio_3.py USER DECISION #3."
            )

        # ------------------------------------------------------------------
        # Duration / projection model — TODO(stable-audio-3): verify whether
        # SA3 keeps the StableAudioProjectionModel-style duration embedding
        # or rolls its own. Until confirmed, leave as None and let the DiT
        # consume raw duration scalars.
        # ------------------------------------------------------------------
        self.projection_model = None  # TODO(stable-audio-3): wire up

        # ------------------------------------------------------------------
        # VAE — SAME autoencoder (ported, see same_autoencoder.py)
        # Variant selection comes from od_config.model_config so users can
        # switch small_music / small_sfx / medium without code changes.
        # ------------------------------------------------------------------
        same_variant = (
            getattr(od_config, "model_config", {}) or {}
        ).get("same_variant", "medium")
        self.vae = SAMEAutoencoder(variant=same_variant, dtype=torch.float32).to(self.device)

        # ------------------------------------------------------------------
        # DiT transformer (weights loaded later by the loader)
        # ------------------------------------------------------------------
        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config, StableAudio3DiTModel,
        )
        self.transformer = StableAudio3DiTModel(od_config=od_config, **transformer_kwargs)

        # ------------------------------------------------------------------
        # Scheduler — TODO(stable-audio-3): confirm whether SA3 ships a custom
        # scheduler or reuses one diffusers exposes. The wrapper lets us slot
        # any standard step()/set_timesteps() scheduler.
        # ------------------------------------------------------------------
        # Placeholder: instantiating with None will fail at forward(); replace
        # with the actual scheduler once upstream is inspected.
        self.scheduler: StableAudio3SchedulerWrapper | None = None

        # Rotary embedding dim — DiT-side detail
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2

        # Cache backend (set by worker if cache-dit / teacache enabled)
        self._cache_backend = None

        # CFG / timestep tracking
        self._guidance_scale: float | None = None
        self._num_timesteps: int | None = None
        self._current_timestep: torch.Tensor | None = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

    # -- standard CFG properties ----------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    # -- validation -----------------------------------------------------------

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        audio_start_in_s: float,
        audio_end_in_s: float,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
    ) -> None:
        if audio_end_in_s < audio_start_in_s:
            raise ValueError(
                f"audio_end_in_s={audio_end_in_s} must be >= audio_start_in_s={audio_start_in_s}"
            )
        if audio_end_in_s - audio_start_in_s > self.max_audio_seconds:
            raise ValueError(
                f"Requested duration {audio_end_in_s - audio_start_in_s:.1f}s exceeds "
                f"SA3 Medium max of {self.max_audio_seconds:.0f}s"
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide only one of `prompt` or `prompt_embeds`.")

    # -- encoding -------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        do_classifier_free_guidance: bool,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text to embeddings.

        Identical contract to SA Open 1.0's encode_prompt — copy that implementation
        and adjust only if SA3 uses a non-T5 encoder (see USER DECISION #3).
        """
        # TODO(stable-audio-3): mirror SA Open 1.0's encode_prompt; key differences,
        # if any, will be (a) tokenizer.model_max_length and (b) projection_model use.
        raise NotImplementedError(
            "encode_prompt: copy from "
            "vllm_omni/diffusion/models/stable_audio/pipeline_stable_audio.py and adapt."
        )

    def encode_duration(
        self,
        audio_start_in_s: float,
        audio_end_in_s: float,
        device: torch.device,
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode duration scalars → conditioning tensors.

        SA Open 1.0 routes this through StableAudioProjectionModel. SA3 may
        or may not — TODO(stable-audio-3): verify and either reuse the
        projection model or implement a raw FourierFeatures embedding here.
        """
        raise NotImplementedError

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_vae: int,
        sample_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample initial latent noise of shape [B, 256, sample_size].

        sample_size is computed per-request from the requested duration (see
        USER DECISION #2 in stable_audio_3_transformer.py).
        """
        from diffusers.utils.torch_utils import randn_tensor

        shape = (batch_size, num_channels_vae, sample_size)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        assert self.scheduler is not None, "scheduler not initialised — see __init__ TODO"
        return latents * self.scheduler.init_noise_sigma

    # -- main entry -----------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        audio_end_in_s: float | None = None,
        audio_start_in_s: float = 0.0,
        guidance_scale: float = 7.0,
        num_waveforms_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str = "np",
    ) -> DiffusionOutput:
        """Generate audio from text prompt(s).

        Once the encode_prompt / encode_duration / scheduler stubs above are
        implemented, this body can be filled in by copying the corresponding
        section from SA Open 1.0's `forward()` and adjusting:

          - num_channels_vae uses self.vae.config.latent_channels (256 for SAME)
          - sample_size is computed from the requested duration, NOT
            self.transformer.config.sample_size (which may be None for SA3)
          - VAE decode goes through SAMEAutoencoder.decode (chunked)
          - rotary_embedding length uses the actual latent_size (variable)
        """
        # TODO(stable-audio-3): port forward() from SA Open 1.0 with the above changes.
        raise NotImplementedError

    # -- weight loading -------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """AutoWeightsLoader entry — delegates to per-component load_weights.

        Note: SA3's DiT defines its own `load_weights` with custom name
        remapping (Pattern 2). AutoWeightsLoader will dispatch DiT-prefixed
        names there automatically.
        """
        return AutoWeightsLoader(self).load_weights(weights)
