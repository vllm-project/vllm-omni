# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoConfig, ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2Tokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import HunyuanVideo15Transformer3DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Extract latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def format_text_input(prompt: list[str], system_message: str) -> list[list[dict[str, str]]]:
    return [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]


def extract_glyph_texts(prompt: str) -> str | None:
    pattern = r'"(.*?)"|' + r"\u201c(.*?)\u201d"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        return ". ".join([f'Text "{text}"' for text in result]) + ". "
    return None


def get_hunyuan_video_15_post_process_func(od_config: OmniDiffusionConfig):
    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(video: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return video
        result = video_processor.postprocess_video(video, output_type=output_type)
        # postprocess_video returns List[List[PIL.Image]] (batch x frames).
        # Flatten to a flat list of PIL Images for the serving endpoint.
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        return result

    return post_process_func


class HunyuanVideo15Pipeline(nn.Module, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = Qwen2_5_VLTextModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )
        t5_config = AutoConfig.from_pretrained(model, subfolder="text_encoder_2", local_files_only=local_files_only)
        self.text_encoder_2 = T5EncoderModel(t5_config, prefix="text_encoder_2").to(dtype=dtype, device=self.device)

        self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # Override the scheduler's shift if flow_shift is explicitly provided.
        # Must set _shift directly since the property has no setter.
        if od_config.flow_shift is not None:
            self.scheduler._shift = od_config.flow_shift

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, HunyuanVideo15Transformer3DModel)
        self.transformer = HunyuanVideo15Transformer3DModel(od_config=od_config, **transformer_kwargs)

        # Check if model uses meanflow (distilled variants)
        self.use_meanflow = getattr(od_config.tf_model_config, "use_meanflow", False)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder_2",
                revision=None,
                prefix="text_encoder_2.",
                fall_back_to_pt=True,
            ),
        ]

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 16
        )
        self.num_channels_latents = self.vae.config.latent_channels if hasattr(self.vae, "config") else 32

        # fmt: off
        # NOTE: whitespace must match the diffusers reference exactly because
        # prompt_template_encode_start_idx=108 depends on the tokenised length of
        # this system message.  The backslash-continuation includes leading spaces.
        self.system_message = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
        # fmt: on
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729
        self.vision_states_dim = 1152

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def _get_mllm_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        num_hidden_layers_to_skip: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_formatted = format_text_input(prompt, self.system_message)

        text_inputs = self.tokenizer.apply_chat_template(
            prompt_formatted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        crop_start = self.prompt_template_encode_start_idx
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        return prompt_embeds.to(dtype=dtype), prompt_attention_mask

    def _get_byte5_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds_list = []
        prompt_embeds_mask_list = []

        for p in prompt:
            glyph_text = extract_glyph_texts(p)

            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, self.tokenizer_2_max_length, self.text_encoder_2.config.d_model),
                    device=device,
                    dtype=dtype,
                )
                glyph_text_embeds_mask = torch.zeros((1, self.tokenizer_2_max_length), device=device, dtype=torch.int64)
            else:
                txt_tokens = self.tokenizer_2(
                    glyph_text,
                    padding="max_length",
                    max_length=self.tokenizer_2_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = self.text_encoder_2(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                glyph_text_embeds = glyph_text_embeds.to(device=device, dtype=dtype)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        return torch.cat(prompt_embeds_list, dim=0), torch.cat(prompt_embeds_mask_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt, device, dtype)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_byte5_prompt_embeds(prompt, device, dtype)

        # Cast masks to model dtype to match diffusers behaviour
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype)

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        negative_prompt_embeds_2 = None
        negative_prompt_embeds_mask_2 = None

        if do_classifier_free_guidance and negative_prompt is not None:
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds, negative_prompt_embeds_mask = self._get_mllm_prompt_embeds(
                negative_prompt, device, dtype
            )
            negative_prompt_embeds_2, negative_prompt_embeds_mask_2 = self._get_byte5_prompt_embeds(
                negative_prompt, device, dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(dtype=dtype)
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(dtype=dtype)
        elif do_classifier_free_guidance:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._get_mllm_prompt_embeds([""], device, dtype)
            negative_prompt_embeds_2, negative_prompt_embeds_mask_2 = self._get_byte5_prompt_embeds([""], device, dtype)
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(dtype=dtype)
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(dtype=dtype)

        return (
            prompt_embeds,
            prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
        )

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            self.num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_cond_latents_and_mask(
        self, latents: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare zero condition latents and mask for T2V mode."""
        batch, channels, frames, height, width = latents.shape
        cond_latents = torch.zeros(batch, channels, frames, height, width, dtype=dtype, device=device)
        mask = torch.zeros(batch, 1, frames, height, width, dtype=dtype, device=device)
        return cond_latents, mask

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 121,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt per request.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        else:
            raise ValueError("Prompt is required for HunyuanVideo-1.5 generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames_val = req.sampling_params.num_frames if req.sampling_params.num_frames else num_frames
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        self._guidance_scale = guidance_scale

        do_cfg = guidance_scale > 1.0

        device = self.device
        dtype = self.transformer.transformer_blocks[0].norm1.linear.weight.dtype

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        (
            prompt_embeds,
            prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=dtype,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
        )

        batch_size = prompt_embeds.shape[0]

        latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames_val,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=req.sampling_params.latents,
        )
        cond_latents, mask = self.prepare_cond_latents_and_mask(latents, dtype, device)

        # Image embeds (zeros for T2V, no mask — let the transformer detect T2V
        # via the all-zeros check, matching the diffusers reference behaviour)
        image_embeds = torch.zeros(
            batch_size,
            self.vision_num_semantic_tokens,
            self.vision_states_dim,
            dtype=dtype,
            device=device,
        )

        # Timesteps — the scheduler handles flow_shift via its config (`shift` param).
        # We just provide unshifted linear sigmas, matching the diffusers reference.
        sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t

                latent_model_input = torch.cat([latents, cond_latents, mask], dim=1)
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                timestep_r = None
                if self.use_meanflow:
                    if i == len(timesteps) - 1:
                        timestep_r = torch.tensor([0.0], device=device)
                    else:
                        timestep_r = timesteps[i + 1]
                    timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype)

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "timestep_r": timestep_r,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_attention_mask": prompt_embeds_mask,
                    "encoder_hidden_states_2": prompt_embeds_2,
                    "encoder_attention_mask_2": prompt_embeds_mask_2,
                    "image_embeds": image_embeds,
                    "return_dict": False,
                }

                negative_kwargs = None
                if do_cfg and negative_prompt_embeds is not None:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "timestep_r": timestep_r,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_attention_mask": negative_prompt_embeds_mask,
                        "encoder_hidden_states_2": negative_prompt_embeds_2,
                        "encoder_attention_mask_2": negative_prompt_embeds_mask_2,
                        "image_embeds": image_embeds,
                        "return_dict": False,
                    }

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg and negative_kwargs is not None,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=req.sampling_params.cfg_normalize,
                )

                latents = self.scheduler_step_maybe_with_cfg(
                    noise_pred,
                    t,
                    latents,
                    do_true_cfg=do_cfg and negative_kwargs is not None,
                )

                pbar.update()

        self._current_timestep = None

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
