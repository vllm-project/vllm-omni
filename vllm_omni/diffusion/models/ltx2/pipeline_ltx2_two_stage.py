# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage entry points for the LTX model family."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, ClassVar

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.lora.request import LoRARequest

from .ltx2_request import LTXRequestInputs
from .pipeline_ltx2 import (
    LTX2ImageToVideoPipeline,
    LTX2Pipeline,
    LTXOneStagePipeline,
)
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline


class LTX2TwoStagesPipeline(nn.Module, SupportsComponentDiscovery):
    """LTX2 two-stage text-to-video entry."""

    dummy_run_num_frames = 2
    supports_request_batch = False
    support_image_input = False

    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["pipe.vae", "pipe.audio_vae"]
    one_stage_pipeline_cls: ClassVar[type[LTXOneStagePipeline]] = LTX2Pipeline

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = od_config.model
        self.distilled = "distilled" in os.path.basename(os.path.normpath(self.model_path))
        if not self.distilled:
            raise NotImplementedError(f"{self.model_path} is not supported for {self.__class__.__name__}.")

        self.pipe = self.one_stage_pipeline_cls(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(vae=self.pipe.vae, od_config=od_config)
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
            )
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
    ) -> DiffusionOutput | list[DiffusionOutput]:
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
            lora_request = LoRARequest(
                lora_name="stage_2_distilled",
                lora_int_id=1,
                lora_path=f"{self.model_path}/ltx-2-19b-distilled-lora-384.safetensors",
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
        del sigmas, return_dict
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
            guidance_scale=(
                getattr(getattr(self.pipe, "one_stage_recipe", None), "guidance_scale", 4.0)
                if guidance_scale is None
                else guidance_scale
            ),
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
        if self.support_image_input:
            image = self.pipe._resolve_request_image(req, image, request_inputs)
        return self._run_two_stage(
            req,
            request_inputs,
            noise_scale=noise_scale,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


class LTX2ImageToVideoTwoStagesPipeline(LTX2TwoStagesPipeline):
    """LTX2 two-stage image-to-video entry."""

    support_image_input = True
    one_stage_pipeline_cls = LTX2ImageToVideoPipeline
