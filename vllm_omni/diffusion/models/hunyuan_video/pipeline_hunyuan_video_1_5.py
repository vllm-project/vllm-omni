# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HunyuanVideo1.5 pipeline adapted for vLLM-Omni.

Notes vs the original diffusers pipeline:
- Components are loaded via vLLM's diffusers loader and executed on the local GPU device.
- Attention/guidance logic uses vLLM's transformer wrapper; PEFT/offload hooks are omitted.
- Adds defaults for optional ByT5/image embeds so integration smoke tests can run with minimal inputs.
"""

import inspect
import os
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2Tokenizer, T5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_1_5_transformer import (
    HunyuanVideo15Transformer3DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

try:
    from diffusers.models.autoencoders import AutoencoderKLHunyuanVideo15
except Exception:  # pragma: no cover - fallback when diffusers is missing class
    from diffusers.models.autoencoders import AutoencoderKL as AutoencoderKLHunyuanVideo15


def format_text_input(prompt: list[str], system_message: str) -> list[list[dict[str, Any]]]:
    """Apply text template for Qwen chat format."""
    template = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]
    return template


def extract_glyph_texts(prompt: str) -> list[str]:
    """Extract quoted substrings for ByT5 conditioning."""
    import re

    pattern = r"\"(.*?)\"|“(.*?)”"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        return result
    return []


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[list[int]] = None,
    sigmas: Optional[list[float]] = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    """Scheduler helper copied from diffusers."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_hunyuan_video_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-process decoded videos to `[0, 1]` float numpy arrays."""

    def post_process_func(videos: torch.Tensor):
        videos = videos.detach().cpu()
        videos = videos.clamp(-1, 1)
        videos = (videos / 2 + 0.5).clamp(0, 1)
        videos = videos.permute(0, 2, 3, 4, 1)  # B, F, H, W, C
        return videos.numpy()

    return post_process_func


class HunyuanVideo15Pipeline(nn.Module):
    """
    Minimal HunyuanVideo1.5 pipeline wired for vLLM execution.

    Differences from the original diffusers pipeline:
    - Loads components via DiffusersPipelineLoader and AutoWeightsLoader (vLLM).
    - Guidance loop simplified for CFG; other guiders can be added later.
    - Offload/PEFT hooks removed; model runs on the selected device.
    - Provides zeroed defaults for missing ByT5/image embeddings to ease smoke tests.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]  # vLLM loader feeds weights directly into our modules

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = Qwen2_5_VLTextModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model, subfolder="text_encoder_2", local_files_only=local_files_only
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )
        self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self._execution_device)
        self.transformer = HunyuanVideo15Transformer3DModel(od_config=od_config)

        self.vae_scale_factor_temporal = getattr(self.vae, "temporal_compression_ratio", 4)
        self.vae_scale_factor_spatial = getattr(self.vae, "spatial_compression_ratio", 16)
        self.target_size = getattr(self.transformer.config, "target_size", 640)
        self.vision_states_dim = getattr(self.transformer.config, "image_embed_dim", 1152)
        self.vision_num_semantic_tokens = 729
        self.num_channels_latents = getattr(self.vae.config, "latent_channels", 32)

        # text prompt template
        self.system_message = (
            "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video."
        )
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.default_num_frames = 121
        self.default_height = 768
        self.default_width = 1360

    @staticmethod
    def _get_mllm_prompt_embeds(
        text_encoder: Qwen2_5_VLTextModel,
        tokenizer: Qwen2Tokenizer,
        prompt: Union[str, list[str]],
        device: torch.device,
        tokenizer_max_length: int = 1000,
        num_hidden_layers_to_skip: int = 2,
        system_message: str | None = None,
        crop_start: int = 108,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt = format_text_input(prompt, system_message or "")

        text_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=tokenizer_max_length + crop_start,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        return prompt_embeds, prompt_attention_mask

    @staticmethod
    def _get_byte5_prompt_embeds(
        tokenizer: ByT5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, list[str]],
        device: torch.device,
        tokenizer_max_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        glyph_texts_nested = [extract_glyph_texts(p) for p in prompt]
        prompt_embeds_list: list[torch.Tensor] = []
        prompt_embeds_mask_list: list[torch.Tensor] = []

        for glyph_text in glyph_texts_nested:
            if len(glyph_text) == 0:
                glyph_text_embeds = torch.zeros(
                    (1, tokenizer_max_length, text_encoder.config.d_model), device=device, dtype=text_encoder.dtype
                )
                glyph_text_embeds_mask = torch.zeros((1, tokenizer_max_length), device=device, dtype=torch.int64)
            else:
                combined_text = ". ".join(glyph_text)
                txt_tokens = tokenizer(
                    combined_text,
                    padding="max_length",
                    max_length=tokenizer_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = text_encoder(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                glyph_text_embeds = glyph_text_embeds.to(device=device)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
    ):
        """Encode prompt with Qwen2.5-VL and ByT5 (glyph extractor) for dual conditioning."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        model_device = next(self.text_encoder.parameters()).device
        model_device_2 = next(self.text_encoder_2.parameters()).device

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=prompt,
                device=model_device,
                tokenizer_max_length=self.tokenizer_max_length,
                system_message=self.system_message,
                crop_start=self.prompt_template_encode_start_idx,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2, prompt_embeds_mask_2 = self._get_byte5_prompt_embeds(
                tokenizer=self.tokenizer_2,
                text_encoder=self.text_encoder_2,
                prompt=prompt,
                device=model_device_2,
                tokenizer_max_length=self.tokenizer_2_max_length,
            )  # ByT5 branch kept for glyph prompts as in HF reference

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_videos_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(batch_size * num_videos_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(batch_size * num_videos_per_prompt, seq_len_2)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        prompt_embeds_2=None,
        prompt_embeds_mask_2=None,
        negative_prompt_embeds_2=None,
        negative_prompt_embeds_mask_2=None,
    ):
        if height is None and width is not None:
            raise ValueError("If `width` is provided, `height` also have to be provided.")
        elif width is None and height is not None:
            raise ValueError("If `height` is provided, `width` also have to be provided.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to "
                "generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to "
                "generate `negative_prompt_embeds`."
            )

        if prompt is None and prompt_embeds_2 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_2`. Cannot leave both `prompt` and `prompt_embeds_2` "
                "undefined."
            )

        if prompt_embeds_2 is not None and prompt_embeds_mask_2 is None:
            raise ValueError(
                "If `prompt_embeds_2` are provided, `prompt_embeds_mask_2` also have to be passed. Make sure to "
                "generate `prompt_embeds_mask_2` from the same text encoder that was used to generate "
                "`prompt_embeds_2`."
            )
        if negative_prompt_embeds_2 is not None and negative_prompt_embeds_mask_2 is None:
            raise ValueError(
                "If `negative_prompt_embeds_2` are provided, `negative_prompt_embeds_mask_2` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask_2` from the same text encoder that was used to "
                "generate `negative_prompt_embeds_2`."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_cond_latents_and_mask(self, latents, dtype: Optional[torch.dtype], device: Optional[torch.device]):
        batch, channels, frames, height, width = latents.shape

        cond_latents_concat = torch.zeros(batch, channels, frames, height, width, dtype=dtype, device=device)
        mask_concat = torch.zeros(batch, 1, frames, height, width, dtype=dtype, device=device)

        return cond_latents_concat, mask_concat

    def _get_latent_model_input(self, latents, cond_latents_concat, mask_concat):
        return torch.cat([latents, cond_latents_concat, mask_concat], dim=1)

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_embeds_mask_2: torch.Tensor,
        negative_prompt_embeds_2: torch.Tensor,
        negative_prompt_embeds_mask_2: torch.Tensor,
        latents: torch.Tensor,
        cond_latents_concat: torch.Tensor,
        mask_concat: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float,
    ):
        """Denoising loop with optional classifier-free guidance (cond + uncond runs)."""
        image_embeds = torch.zeros(
            latents.shape[0],
            self.vision_num_semantic_tokens,
            self.vision_states_dim,
            dtype=self.transformer.proj_out.weight.dtype,
            device=latents.device,
        )  # HF i2v uses vision tokens; zero here keeps t2v path lightweight for tests

        for i, t in enumerate(timesteps):
            latent_model_input = self._get_latent_model_input(latents, cond_latents_concat, mask_concat)
            timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_embeds_mask,
                encoder_hidden_states_2=prompt_embeds_2,
                encoder_attention_mask_2=prompt_embeds_mask_2,
                image_embeds=image_embeds,
                attention_kwargs={},
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_attention_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states_2=negative_prompt_embeds_2,
                    encoder_attention_mask_2=negative_prompt_embeds_mask_2,
                    image_embeds=image_embeds,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: Union[str, list[str]] = "",
        negative_prompt: Union[str, list[str], None] = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int = 50,
        sigmas: Optional[list[float]] = None,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
    ) -> DiffusionOutput:
        """End-to-end text-to-video call: encode, sample latents, denoise, decode."""
        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt
        height = req.height or height or self.default_height
        width = req.width or width or self.default_width
        num_frames = (
            int(req.num_frames) if getattr(req, "num_frames", None) else (num_frames or self.default_num_frames)
        )
        num_inference_steps = req.num_inference_steps or num_inference_steps
        generator = req.generator or generator
        guidance_scale = req.guidance_scale if getattr(req, "guidance_scale", None) is not None else guidance_scale
        sigmas = req.sigmas if getattr(req, "sigmas", None) is not None else sigmas
        req_num_outputs = getattr(req, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_videos_per_prompt = req_num_outputs

        height = int(height // self.vae_scale_factor_spatial) * self.vae_scale_factor_spatial
        width = int(width // self.vae_scale_factor_spatial) * self.vae_scale_factor_spatial

        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt is not None

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = self.encode_prompt(
            prompt=prompt,
            device=self._execution_device,
            dtype=self.transformer.proj_out.weight.dtype,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
        )

        if do_classifier_free_guidance:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                device=self._execution_device,
                dtype=self.transformer.proj_out.weight.dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            )
        else:
            negative_prompt_embeds = prompt_embeds
            negative_prompt_embeds_mask = prompt_embeds_mask
            negative_prompt_embeds_2 = prompt_embeds_2
            negative_prompt_embeds_mask_2 = prompt_embeds_mask_2

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            self.num_channels_latents,
            height,
            width,
            num_frames,
            self.transformer.proj_out.weight.dtype,
            self._execution_device,
            generator,
            latents,
        )
        cond_latents_concat, mask_concat = self.prepare_cond_latents_and_mask(
            latents, self.transformer.proj_out.weight.dtype, self._execution_device
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device=self._execution_device, sigmas=sigmas
        )
        self._num_timesteps = len(timesteps)

        latents = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
            latents,
            cond_latents_concat,
            mask_concat,
            timesteps,
            guidance_scale,
        )

        latents = latents.to(self.vae.dtype) / getattr(self.vae.config, "scaling_factor", 0.18215)
        video = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=video)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
