# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, Qwen3VLModel
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2Transformer2DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

logger = logging.getLogger(__name__)

# Text-encoder layer taps used when model_index.json does not specify ``text_encoder_select_layers``.
DEFAULT_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def get_krea2_post_process_func(od_config: OmniDiffusionConfig):
    model_name = od_config.model
    # Read only the VAE config (``get_hf_file_to_dict`` resolves a local dir via
    # ``Path(model)/file`` and a hub repo via a single-file download) instead of
    # pulling all weights just to compute ``vae_scale_factor``.
    vae_config = get_hf_file_to_dict("vae/config.json", model_name) or {}
    vae_scale_factor = 2 ** len(vae_config["temperal_downsample"]) if "temperal_downsample" in vae_config else 8

    patch_size = 2
    try:
        model_index = get_hf_file_to_dict("model_index.json", model_name) or {}
        patch_size = int(model_index.get("patch_size", patch_size))
    except Exception:  # noqa: BLE001
        pass

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * patch_size)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
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


class Krea2Pipeline(nn.Module, DiffusionPipelineProfilerMixin, ProgressBarMixin, SupportsComponentDiscovery):
    """The Krea 2 text-to-image pipeline (vLLM-Omni port of ``diffusers.Krea2Pipeline``).

    Components:
    - ``scheduler``: ``FlowMatchEulerDiscreteScheduler`` (resolution-aware exponential time shift).
    - ``vae``: the Qwen-Image VAE (f8, 16 latent channels).
    - ``text_encoder``: a Qwen3-VL model; the pipeline consumes a stack of hidden states tapped from several
      decoder layers rather than the last hidden state.
    - ``tokenizer``: the tokenizer paired with the text encoder.
    - ``transformer``: the Krea 2 single-stream MMDiT that predicts the flow-matching velocity.
    """

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    _PROFILER_TARGETS: ClassVar[list[str]] = ["text_encoder.forward", "diffuse", "vae.decode"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        # Only the transformer goes through the standard weights loader; text encoder and VAE load eagerly below.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.isdir(model)

        # Pipeline-level config lives in model_index.json, not the transformer config.
        model_index = get_hf_file_to_dict("model_index.json", model) or {}
        select_layers = model_index.get("text_encoder_select_layers") or DEFAULT_TEXT_ENCODER_SELECT_LAYERS
        self.text_encoder_select_layers = tuple(int(i) for i in select_layers)
        self.is_distilled = bool(model_index.get("is_distilled", False))
        self.patch_size = int(model_index.get("patch_size", 2))

        subfolders = ["scheduler", "text_encoder", "vae", "tokenizer"]
        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3VLModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
        )
        # Drop the unused Qwen3-VL vision tower before moving to GPU so it never consumes GPU memory.
        if hasattr(self.text_encoder, "visual"):
            del self.text_encoder.visual
        else:
            logger.warning("Krea2: vision tower not found on text encoder; skipping drop")
        self.text_encoder = self.text_encoder.to(self.device)

        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLQwenImage.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
        ).to(self.device)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, Krea2Transformer2DModel)
        self.transformer = Krea2Transformer2DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)
        self.default_sample_size = 1024

        # Chat template tokenized as a fixed-length block: prompt padded to a fixed length first, assistant suffix
        # appended after the padding; the first ``prompt_template_encode_start_idx`` system tokens are later dropped.
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_num_suffix_tokens = 5

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def get_text_hidden_states(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize ``prompt`` into the fixed-length Krea 2 layout and tap the selected encoder hidden states.

        Returns ``(hidden_states, attention_mask)`` of shapes ``(batch, text_seq_len, num_text_layers,
        text_hidden_dim)`` and ``(batch, text_seq_len)`` (bool).
        """
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + e for e in prompt]
        text_tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - self.prompt_template_encode_num_suffix_tokens,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = self.tokenizer([self.prompt_template_encode_suffix] * len(text), return_tensors="pt").to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        # Padding sits in the middle (``[prefix | prompt | PAD | suffix]``), so positions must count only real tokens
        # (padding consumes no position). Broadcast across the 3 mRoPE axes (T/H/W equal for text).
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2)

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.device

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.get_text_hidden_states(prompt, max_sequence_length, device)

        batch_size, seq_len, num_text_layers, dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, num_text_layers, dim)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        p = self.patch_size
        latents = latents.view(batch_size, num_channels_latents, height // p, p, width // p, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // p) * (width // p), num_channels_latents * p * p)
        return latents

    def _unpack_latents(self, latents, height, width):
        batch_size, _, channels = latents.shape
        p = self.patch_size

        height = p * (int(height) // (self.vae_scale_factor * p))
        width = p * (int(width) // (self.vae_scale_factor * p))

        latents = latents.view(batch_size, height // p, width // p, channels // (p * p), p, p)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (p * p), 1, height, width)
        return latents

    @staticmethod
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device):
        """Build the ``(text_seq_len + grid_height * grid_width, 3)`` rotary coordinates: text tokens sit at the
        origin, image tokens carry their ``(0, h, w)`` latent-grid coordinates."""
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return torch.cat([text_ids, image_ids], dim=0)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, latent_height, latent_width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)
        return latents

    def _extract_prompts(self, prompts):
        """Extract prompt and negative_prompt from the request's OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the flow-matching denoising loop and return the final latents."""
        self.scheduler.set_begin_index(0)
        transformer_dtype = self.transformer.dtype
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for _, t in enumerate(timesteps):
                timestep = (t / self.scheduler.config.num_train_timesteps).expand(latents.shape[0])
                timestep = timestep.to(transformer_dtype)
                latent_input = latents.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_input,
                    encoder_hidden_states=prompt_embeds.to(transformer_dtype),
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_embeds_mask,
                )

                if self.do_classifier_free_guidance:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_input,
                        encoder_hidden_states=negative_prompt_embeds.to(transformer_dtype),
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=negative_prompt_embeds_mask,
                    )
                    noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pil",
        max_sequence_length: int = 512,
    ) -> DiffusionOutput:
        extracted_prompt, extracted_negative = self._extract_prompts(req.prompts)
        prompt = extracted_prompt or prompt
        if extracted_negative is not None:
            negative_prompt = extracted_negative

        sp = req.sampling_params
        height = sp.height or height
        width = sp.width or width
        num_inference_steps = sp.num_inference_steps or num_inference_steps
        sigmas = sp.sigmas or sigmas
        max_sequence_length = sp.max_sequence_length or max_sequence_length
        generator = sp.generator or generator
        if sp.guidance_scale_provided:
            guidance_scale = sp.guidance_scale
        else:
            # Distilled checkpoint is trained guidance-free (and the request layer coerces an explicit
            # guidance_scale=0 to "not provided"), so default it to no CFG.
            guidance_scale = 0.0 if self.is_distilled else guidance_scale
        num_images_per_prompt = sp.num_outputs_per_prompt if sp.num_outputs_per_prompt > 0 else num_images_per_prompt

        multiple = self.vae_scale_factor * self.patch_size
        if height % multiple != 0 or width % multiple != 0:
            rounded_height = ((height + multiple - 1) // multiple) * multiple
            rounded_width = ((width + multiple - 1) // multiple) * multiple
            logger.warning(
                "`height` and `width` must be multiples of %d; rounding up from %dx%d to %dx%d.",
                multiple,
                height,
                width,
                rounded_height,
                rounded_width,
            )
            height, width = rounded_height, rounded_width

        device = self.device
        self._guidance_scale = guidance_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels // (self.patch_size**2)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        grid_height = height // (self.vae_scale_factor * self.patch_size)
        grid_width = width // (self.vae_scale_factor * self.patch_size)
        position_ids = self.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, device)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        if self.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )

        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            position_ids=position_ids,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            guidance_scale=guidance_scale,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        # VAE and text encoder load eagerly in __init__; record their params so strict load checks pass.
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        if self.text_encoder is not None:
            loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights
