# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One-stage entry points for the LTX model family."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from diffusers.utils.torch_utils import randn_tensor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    LTX23_COMPONENT_PROFILE,
    LTXComponentProfile,
    initialize_pipeline_components,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_guidance import (
    LTX_LEGACY_VELOCITY_GUIDANCE,
    LTX_OFFICIAL_X0_GUIDANCE,
    LTXGuidanceStrategy,
)
from .ltx2_pipeline_base import LTXPipelineBase
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE, LTX23_ONE_STAGE_RECIPE, LTXOneStageRecipe


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
    decode_noise_scale_values = (
        decode_timestep_values
        if decode_noise_scale is None
        else _expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )


class LTXOneStagePipeline(LTXPipelineBase, DiffusionPipelineProfilerMixin):
    """Single execution path configured by model-version and task entries."""

    component_profile: ClassVar[LTXComponentProfile]
    guidance_strategy: ClassVar[LTXGuidanceStrategy]
    one_stage_recipe: ClassVar[LTXOneStageRecipe]

    supports_request_batch = True
    supports_guidance_rescale = False
    connector_batches_cfg = False
    distributed_video_decode = True
    support_image_input = False
    dummy_run_num_frames = 2

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        del prefix
        super().__init__()
        initialize_pipeline_components(self, od_config)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

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
            return self._make_output((latents, audio_latents))

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

        dist_initialized = torch.distributed.is_initialized()
        is_output_rank = not dist_initialized or torch.distributed.get_rank() == 0
        vae_decode_needs_all_ranks = False
        is_distributed_vae_enabled = getattr(self.vae, "is_distributed_enabled", None)
        if self.distributed_video_decode and dist_initialized and callable(is_distributed_vae_enabled):
            try:
                # Distributed tiled decode is collective, so every rank must enter it.
                vae_decode_needs_all_ranks = bool(is_distributed_vae_enabled())
            except Exception:
                pass

        should_decode_video = not self.distributed_video_decode or is_output_rank or vae_decode_needs_all_ranks
        if should_decode_video:
            video = self.vae.decode(latents.to(self.vae.dtype), timestep_decode, return_dict=False)[0]
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if self.distributed_video_decode and not is_output_rank:
            return self._make_output(
                (
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                )
            )

        if video.numel() > 0:
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        generated_mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        audio = self.vocoder(generated_mel)
        return self._make_output((video, audio))

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        image: Any | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
        guidance_rescale: float = 0.0,
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
    ) -> DiffusionOutput | list[DiffusionOutput]:
        del return_dict
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
            guidance_scale=self.one_stage_recipe.guidance_scale if guidance_scale is None else guidance_scale,
            guidance_rescale=guidance_rescale,
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
        image = self._resolve_request_image(req, image, request_inputs)
        forward_kwargs = {
            "noise_scale": noise_scale,
            "sigmas": sigmas,
            "timesteps": timesteps,
            "attention_kwargs": attention_kwargs,
        }
        if self.support_image_input:
            forward_kwargs["image"] = image
        return self._forward_impl(
            req,
            request_inputs,
            **forward_kwargs,
        )


class LTX2Pipeline(LTXOneStagePipeline):
    """LTX2 one-stage text-to-video entry."""

    supports_guidance_rescale = True
    component_profile = LTX2_COMPONENT_PROFILE
    guidance_strategy = LTX_LEGACY_VELOCITY_GUIDANCE
    one_stage_recipe = LTX2_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)


class LTX23Pipeline(LTXOneStagePipeline):
    """LTX2.3 one-stage text-to-video entry."""

    connector_batches_cfg = True
    preserve_sp_padded_audio_duration = True
    scheduler_shift_uses_max_sequence_length = True
    reports_stage_durations = True
    component_profile = LTX23_COMPONENT_PROFILE
    guidance_strategy = LTX_OFFICIAL_X0_GUIDANCE
    one_stage_recipe = LTX23_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)


class LTX2ImageToVideoPipeline(LTXI2VConditioningMixin, LTX2Pipeline):
    """LTX2 one-stage image-to-video entry."""


class LTX23ImageToVideoPipeline(LTXI2VConditioningMixin, LTX23Pipeline):
    """LTX2.3 one-stage image-to-video entry."""


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX2 T2V entry for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()


class LTX2I2VDMD2Pipeline(DMD2PipelineMixin, LTX2ImageToVideoPipeline):
    """LTX2 I2V entry for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
