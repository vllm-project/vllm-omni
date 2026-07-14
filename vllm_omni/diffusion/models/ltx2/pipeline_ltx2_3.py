# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 entry points built on the shared LTX model-family runtime."""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import torch
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_components import (
    LTX23_COMPONENT_PROFILE,
    initialize_pipeline_components,
)
from .ltx2_guidance import LTX_OFFICIAL_X0_GUIDANCE
from .ltx2_pipeline_base import LTXPipelineBase
from .ltx2_recipes import LTX23_ONE_STAGE_RECIPE

logger = init_logger(__name__)


def _is_output_rank() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _vae_decode_needs_all_ranks(vae: Any) -> bool:
    if not torch.distributed.is_initialized():
        return False
    is_distributed_enabled = getattr(vae, "is_distributed_enabled", None)
    if not callable(is_distributed_enabled):
        return False
    try:
        return bool(is_distributed_enabled())
    except Exception:
        return False


def _should_decode_video_on_rank(vae: Any) -> bool:
    return _is_output_rank() or _vae_decode_needs_all_ranks(vae)


def _detect_vocoder_output_sample_rate(model: str) -> int | None:
    """Detect the vocoder output sample rate from vocoder/config.json.

    This runs at factory time (engine process) so the rate is captured in
    the post-process closure and doesn't need cross-process communication.

    Returns:
        Output sample rate (e.g. 48000 for LTX-2.3 BWE vocoder) or None.
    """
    vocoder_config_path = os.path.join(model, "vocoder", "config.json")
    if not os.path.exists(vocoder_config_path):
        try:
            vocoder_config_path = hf_hub_download(model, "vocoder/config.json")
        except Exception:
            return None
    try:
        with open(vocoder_config_path) as f:
            cfg = json.load(f)
        return cfg.get("output_sampling_rate")
    except Exception:
        return None


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    """Factory for the LTX-2.3 post-process function.

    Detects the vocoder output sample rate at factory time and captures it
    in the closure so that the audio_sample_rate flows through
    DiffusionEngine -> OmniRequestOutput -> serving_video.
    """
    output_sr = _detect_vocoder_output_sample_rate(od_config.model)

    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            result: dict[str, Any] = {"video": video, "audio": audio}
            if output_sr is not None:
                result["audio_sample_rate"] = output_sr
            return result
        return output

    return post_process_func


def _expand_per_prompt_decode_value(
    value: float | list[float],
    *,
    prompt_batch_size: int,
    effective_batch_size: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list):
        return [value] * effective_batch_size
    if len(value) == 1:
        return value * effective_batch_size
    if len(value) == effective_batch_size:
        return value
    if prompt_batch_size > 0 and len(value) == prompt_batch_size and effective_batch_size % prompt_batch_size == 0:
        repeats = effective_batch_size // prompt_batch_size
        return [item for item in value for _ in range(repeats)]
    raise ValueError(
        f"`{field_name}` must have length 1, prompt batch size ({prompt_batch_size}), or effective batch size"
        f" ({effective_batch_size}); got {len(value)}."
    )


def _prepare_decode_timestep_conditioning(
    *,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    prompt_batch_size: int,
    effective_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_timestep_values = _expand_per_prompt_decode_value(
        decode_timestep,
        prompt_batch_size=prompt_batch_size,
        effective_batch_size=effective_batch_size,
        field_name="decode_timestep",
    )
    if decode_noise_scale is None:
        decode_noise_scale_values = decode_timestep_values
    else:
        decode_noise_scale_values = _expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )


class LTX23Pipeline(LTXPipelineBase, DiffusionPipelineProfilerMixin):
    """LTX-2.3 one-stage configuration of the shared LTX pipeline.

    Version-specific behavior is limited to:
    - LTX-2.3 component/profile selection, including the 48 kHz BWE vocoder
    - batched request and connector execution
    - official x0-space guidance strategy
    """

    supports_request_batch = True
    connector_batches_cfg = True
    # Audio is diffused jointly with video; warmup must size audio tokens.
    dummy_run_num_frames = 2
    component_profile = LTX23_COMPONENT_PROFILE
    guidance_strategy = LTX_OFFICIAL_X0_GUIDANCE
    one_stage_recipe = LTX23_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        initialize_pipeline_components(self, od_config)

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------
    # Text Encoding (LTX-2.3 specific)
    # ------------------------------------------------------------------

    def _resolve_audio_latent_length(self, audio_latent_length: int, audio_latents: torch.Tensor | None) -> int:
        if audio_latents is None or audio_latents.ndim != 4:
            return audio_latent_length

        provided_latent_length = audio_latents.shape[2]
        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_latent_length = self._get_sp_padded_audio_latent_length(audio_latent_length, int(sp_size))

        # Keep requested duration semantics when callers pass 4D latents that
        # are already padded for SP; other 4D lengths retain shape inference.
        if provided_latent_length in {audio_latent_length, padded_latent_length}:
            return audio_latent_length
        return provided_latent_length

    def _decode_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        output_type: str,
        connector_prompt_embeds: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        prompt_batch_size: int,
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(
                output=(latents, audio_latents),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep_decode, decode_noise_scale_t = _prepare_decode_timestep_conditioning(
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                prompt_batch_size=prompt_batch_size,
                effective_batch_size=latents.shape[0],
                device=device,
                dtype=latents.dtype,
            )
            latents = (1 - decode_noise_scale_t) * latents + decode_noise_scale_t * noise

        if _should_decode_video_on_rank(self.vae):
            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep_decode, return_dict=False)[0]
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if not _is_output_rank():
            return DiffusionOutput(
                output=(
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                ),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        if video.numel() > 0:
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        audio_latents = audio_latents.to(self.audio_vae.dtype)
        generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = self.vocoder(generated_mel_spectrograms)

        return DiffusionOutput(
            output=(video, audio),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _scheduler_shift_sequence_length(
        self,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> int:
        # Diffusers LTX2Pipeline hardcodes max_image_seq_len for this path.
        return self.scheduler.config.get("max_image_seq_len", 4096)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = LTX23_ONE_STAGE_RECIPE.guidance_scale,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> list[DiffusionOutput]:
        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )
        return self._forward_impl(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
        )
