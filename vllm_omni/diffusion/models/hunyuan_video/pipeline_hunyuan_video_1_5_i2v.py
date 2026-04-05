# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import (
    AutoConfig,
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import HunyuanVideo15Transformer3DModel
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    extract_glyph_texts,
    format_text_input,
    get_hunyuan_video_15_post_process_func,
    retrieve_latents,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def get_hunyuan_video_15_i2v_post_process_func(od_config: OmniDiffusionConfig):
    return get_hunyuan_video_15_post_process_func(od_config)


def get_hunyuan_video_15_i2v_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process function for I2V: load and resize image."""

    max_area = 480 * 832
    divisor = 16  # Must be divisible by VAE spatial compression

    def pre_process_func(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if not req.prompts:
            return req

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            return req

        multi_modal_data = prompt_data.get("multi_modal_data", {})
        raw_image = multi_modal_data.get("image", None)

        if raw_image is None:
            return req

        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            return req

        height = req.sampling_params.height
        width = req.sampling_params.width

        if height is None or width is None:
            w, h = image.size
            aspect = w / h
            target_h = int((max_area / aspect) ** 0.5)
            target_w = int(target_h * aspect)
            height = height or ((target_h + divisor - 1) // divisor * divisor)
            width = width or ((target_w + divisor - 1) // divisor * divisor)
            req.sampling_params.height = height
            req.sampling_params.width = width

        return req

    return pre_process_func


class HunyuanVideo15I2VPipeline(
    nn.Module, CFGParallelMixin, SupportImageInput, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    support_image_input = True
    color_format = "RGB"

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

        self.image_encoder = SiglipVisionModel.from_pretrained(
            model, subfolder="image_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            model, subfolder="feature_extractor", local_files_only=local_files_only
        )

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
    ) -> tuple:
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

    def _get_image_embeds(
        self,
        image: PIL.Image.Image,
        device: torch.device,
    ) -> torch.Tensor:
        """Get semantic image embeddings from SigLIP."""
        image_encoder_dtype = next(self.image_encoder.parameters()).dtype
        pixel_values = self.feature_extractor(images=image, do_resize=True, return_tensors="pt").pixel_values.to(
            device=device, dtype=image_encoder_dtype
        )
        image_enc_hidden_states = self.image_encoder(pixel_values).last_hidden_state
        return image_enc_hidden_states

    def _get_image_latents(
        self,
        image: PIL.Image.Image,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode image to VAE latents for first-frame conditioning."""
        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        image_tensor = video_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=self.vae.dtype)  # [B, C, 1, H, W]
        image_latents = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
        image_latents = image_latents * self.vae.config.scaling_factor
        return image_latents

    def prepare_cond_latents_and_mask(
        self,
        latents: torch.Tensor,
        image: PIL.Image.Image,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare condition latents and mask for I2V: first frame = image, rest = zeros."""
        batch, channels, frames, lat_height, lat_width = latents.shape

        image_latents = self._get_image_latents(image, height, width, device)

        # Repeat image latent across all frames, then zero out non-first frames
        latent_condition = image_latents.repeat(batch, 1, frames, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        latent_condition = latent_condition.to(device=device, dtype=dtype)

        # Mask: 1.0 for first frame (conditioned), 0.0 for rest (to denoise)
        latent_mask = torch.zeros(batch, 1, frames, lat_height, lat_width, dtype=dtype, device=device)
        latent_mask[:, :, 0, :, :] = 1.0

        return latent_condition, latent_mask

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

        num_channels = self.num_channels_latents
        shape = (
            batch_size,
            num_channels,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

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
            raise ValueError("Prompt is required for HunyuanVideo-1.5 I2V generation.")

        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else {}
        raw_image = multi_modal_data.get("image", None)
        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                logger.warning("Received a list of images. Only the first image will be used.")
            raw_image = raw_image[0]

        if raw_image is None:
            raise ValueError(
                "Image is required for HunyuanVideo-1.5 I2V generation. Pass it via multi_modal_data={'image': <image>}"
            )

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            image = cast(PIL.Image.Image, raw_image)

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

        # Get image embeddings from SigLIP (no explicit mask — the transformer
        # detects I2V via non-zero image_embeds, matching diffusers behaviour)
        image_embeds = self._get_image_embeds(image, device).to(dtype=dtype)

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

        cond_latents, mask = self.prepare_cond_latents_and_mask(latents, image, height, width, dtype, device)

        # Timesteps — the scheduler handles flow_shift via its config (`shift` param).
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
                    # For I2V CFG, negative still uses image embeds (only text is unconditional)
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
