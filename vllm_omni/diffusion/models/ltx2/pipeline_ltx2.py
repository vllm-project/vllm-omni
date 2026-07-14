# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, ClassVar

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.lora.request import LoRARequest

from .ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    initialize_pipeline_components,
)
from .ltx2_guidance import LTX_LEGACY_VELOCITY_GUIDANCE
from .ltx2_pipeline_base import LTXPipelineBase
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE
from .ltx2_request import LTXRequestInputs
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

logger = init_logger(__name__)


def get_ltx2_post_process_func(
    od_config: OmniDiffusionConfig,
):
    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            return {"video": video, "audio": audio}
        return output

    return post_process_func


class LTX2Pipeline(LTXPipelineBase):
    supports_request_batch = False
    supports_guidance_rescale = True

    component_profile = LTX2_COMPONENT_PROFILE
    guidance_strategy = LTX_LEGACY_VELOCITY_GUIDANCE
    one_stage_recipe = LTX2_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)

    # Audio is diffused jointly with video; warmup must size audio tokens.
    dummy_run_num_frames = 2

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        initialize_pipeline_components(self, od_config)

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
            return DiffusionOutput(output=(latents, audio_latents))

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )
            decode_timestep_values = decode_timestep
            if not isinstance(decode_timestep_values, list):
                decode_timestep_values = [decode_timestep_values] * prompt_batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep_values
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * prompt_batch_size

            timestep_decode = torch.tensor(
                decode_timestep_values,
                device=device,
                dtype=latents.dtype,
            )
            decode_noise_scale_tensor = torch.tensor(
                decode_noise_scale,
                device=device,
                dtype=latents.dtype,
            )[:, None, None, None, None]
            latents = (1 - decode_noise_scale_tensor) * latents + decode_noise_scale_tensor * noise

        video = self.vae.decode(latents.to(self.vae.dtype), timestep_decode, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        generated_mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        audio = self.vocoder(generated_mel)
        return DiffusionOutput(output=(video, audio))

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
        guidance_scale: float = LTX2_ONE_STAGE_RECIPE.guidance_scale,
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
    ) -> DiffusionOutput:
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
        return self._forward_impl(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
        )


class LTX2TwoStagesPipeline(nn.Module, SupportsComponentDiscovery):
    """LTX2TwoStagesPipeline is for two stages image to video generation"""

    dummy_run_num_frames = 2
    supports_request_batch = False

    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["pipe.vae", "pipe.audio_vae"]
    one_stage_pipeline_cls = LTX2Pipeline

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = od_config.model
        self.distilled = False
        # User provided model path may contain '/' in the end and basename function
        # will not return the expected directory name, so we need to remove it by normpath
        if "distilled" in os.path.basename(os.path.normpath(self.model_path)):
            self.distilled = True
        else:
            raise NotImplementedError(f"{self.model_path} is not supported for {self.__class__.__name__}.")

        self.pipe = self.one_stage_pipeline_cls(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae,
            od_config=od_config,
        )

        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.pipe,
            device=self.device,
            dtype=self.dtype,
            max_cached_adapters=od_config.max_cpu_loras,
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="pipe.transformer.",
                fall_back_to_pt=True,
            ),
        ]

    def _run_two_stage(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        image: Any | None = None,
    ) -> DiffusionOutput:
        stage1 = self.pipe._run_denoise_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )

        upscaled_video_latent = self.upsample_pipe(
            latents=stage1.video,
            output_type="latent",
            return_dict=False,
        )[0]

        if not self.distilled:
            lora_path = f"{self.model_path}/ltx-2-19b-distilled-lora-384.safetensors"
            lora_request = LoRARequest(
                lora_name="stage_2_distilled",
                lora_int_id=1,
                lora_path=lora_path,
            )
            self.lora_manager.set_active_adapter(lora_request, lora_scale=1.0)
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )

        stage2_inputs = replace(
            request_inputs,
            num_inference_steps=3,
            guidance_scale=1.0,
            latents=upscaled_video_latent,
            audio_latents=stage1.audio,
            output_type="np",
        )
        stage2 = self.pipe._run_denoise_phase(
            req,
            stage2_inputs,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            timesteps=None,
            attention_kwargs=attention_kwargs,
            prompt_context=stage1.forward_context.prompt_context,
        )
        return self.pipe._decode_and_split(stage2.forward_context, stage2.video, stage2.audio)

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
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
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
    ) -> DiffusionOutput:
        request_inputs = self.pipe._resolve_request_inputs(
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
        return self._run_two_stage(
            req,
            request_inputs,
            noise_scale=noise_scale,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX-2 T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
