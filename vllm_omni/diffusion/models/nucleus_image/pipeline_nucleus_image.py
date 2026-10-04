# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/huggingface/diffusers.

import copy
import inspect
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.nucleus_image.nucleus_image_transformer import (
    NucleusMoEImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

logger = init_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an image generation assistant. Follow the user's prompt literally. "
    "Pay careful attention to spatial layout: objects described as on the left must appear on the left, "
    "on the right on the right. Match exact object counts and assign colors to the correct objects."
)


def get_nucleus_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


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


def _cfg_normalize_function(positive_noise_pred: torch.Tensor, combined_noise_pred: torch.Tensor) -> torch.Tensor:
    cond_norm = torch.norm(positive_noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(combined_noise_pred, dim=-1, keepdim=True)
    return combined_noise_pred * (cond_norm / noise_norm)


def _combine_cfg_noise(
    positive_noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
    cfg_normalize: bool = False,
) -> torch.Tensor:
    combined = negative_noise_pred + true_cfg_scale * (positive_noise_pred - negative_noise_pred)
    if cfg_normalize:
        combined = _cfg_normalize_function(positive_noise_pred, combined)
    return combined


def _should_do_true_cfg(
    true_cfg_scale: float,
) -> bool:
    return true_cfg_scale > 1


class NucleusMoEImagePipeline(nn.Module, DiffusionPipelineProfilerMixin):
    supports_step_execution: ClassVar[bool] = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        ).to(self.device)
        self.vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, NucleusMoEImageTransformer2DModel)
        self.transformer = NucleusMoEImageTransformer2DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )

        self.stage = None

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.default_sample_size = 128
        self.default_max_sequence_length = 1024
        self.default_return_index = -8
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        return_index=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both undefined.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )

        if return_index is not None and abs(return_index) >= self.text_encoder.config.text_config.num_hidden_layers:
            raise ValueError(
                f"absolute value of `return_index` cannot be >= "
                f"{self.text_encoder.config.text_config.num_hidden_layers} "
                f"but is {abs(return_index)}"
            )

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode_prompt(
        self,
        prompt: str | list[str] = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int | None = None,
        return_index: int | None = None,
    ):
        device = device or self._execution_device
        if return_index is None:
            return_index = self.default_return_index
        max_sequence_length = max_sequence_length or self.default_max_sequence_length

        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            formatted = [self._format_prompt(p) for p in prompt]

            inputs = self.processor(
                text=formatted,
                padding="longest",
                pad_to_multiple_of=8,
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device=device)

            prompt_embeds_mask = inputs.attention_mask

            outputs = self.text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.hidden_states[return_index]
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(device=device)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to(device=device)

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, patch_size):
        latents = latents.view(
            batch_size, num_channels_latents, height // patch_size, patch_size, width // patch_size, patch_size
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // patch_size) * (width // patch_size), num_channels_latents * patch_size * patch_size
        )
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, patch_size, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = patch_size * (int(height) // (vae_scale_factor * patch_size))
        width = patch_size * (int(width) // (vae_scale_factor * patch_size))
        latents = latents.view(
            batch_size,
            height // patch_size,
            width // patch_size,
            channels // (patch_size * patch_size),
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (patch_size * patch_size), 1, height, width)
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        patch_size,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = patch_size * (int(height) // (self.vae_scale_factor * patch_size))
        width = patch_size * (int(width) // (self.vae_scale_factor * patch_size))
        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width, patch_size)
        return latents

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _extract_prompts(self, prompts):
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    def _prepare_generation_context(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        sigmas,
        guidance_scale,
        num_images_per_prompt,
        generator,
        max_sequence_length,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        latents=None,
        attention_kwargs=None,
        callback_on_step_end_tensor_inputs=None,
        return_index=None,
    ):
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
            return_index,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False
        self.transformer.reset_text_kv_cache()

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_cfg = _should_do_true_cfg(guidance_scale)
        if do_cfg and not has_neg_prompt:
            negative_prompt = [""] * batch_size

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            return_index=return_index,
        )

        if do_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                return_index=return_index,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        patch_size = self.transformer.patch_size
        num_channels_latents = self.transformer.in_channels // (patch_size * patch_size)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            patch_size,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        img_shapes = [
            (1, height // self.vae_scale_factor // patch_size, width // self.vae_scale_factor // patch_size)
        ] * (batch_size * num_images_per_prompt)

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps,
            sigmas,
            latents.shape[1],
        )
        self._num_timesteps = len(timesteps)
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "do_true_cfg": do_cfg,
            "true_cfg_scale": guidance_scale,
            "latents": latents,
            "img_shapes": img_shapes,
            "timesteps": timesteps,
        }

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        patch_size = self.transformer.patch_size
        latents = self._unpack_latents(latents, height, width, patch_size, self.vae_scale_factor)
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

    def prepare_encode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> "DiffusionRequestState":
        sampling = state.sampling
        prompt, negative_prompt = self._extract_prompts(state.prompts or [])

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=sampling.num_inference_steps or 50,
            sigmas=sampling.sigmas,
            guidance_scale=sampling.guidance_scale if sampling.guidance_scale_provided else 1.0,
            num_images_per_prompt=sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1,
            generator=sampling.generator,
            max_sequence_length=sampling.max_sequence_length or self.default_max_sequence_length,
            attention_kwargs=kwargs.get("attention_kwargs"),
            return_index=self.default_return_index,
        )

        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        state.prompt_embeds = ctx["prompt_embeds"]
        state.prompt_embeds_mask = ctx["prompt_embeds_mask"]
        state.negative_prompt_embeds = ctx["negative_prompt_embeds"]
        state.negative_prompt_embeds_mask = ctx["negative_prompt_embeds_mask"]
        state.do_true_cfg = ctx["do_true_cfg"]
        state.true_cfg_scale = ctx["true_cfg_scale"]
        state.latents = ctx["latents"]
        state.timesteps = ctx["timesteps"]
        state.step_index = 0
        state.scheduler = req_scheduler
        state.img_shapes = ctx["img_shapes"]

        return state

    def _predict_noise(
        self,
        latents: torch.Tensor,
        img_shapes: list,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        noise_pred = self.transformer(
            hidden_states=latents,
            img_shapes=img_shapes,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=timestep / self.scheduler.config.num_train_timesteps,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    def denoise_step(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        if self.interrupt:
            return None

        t = state.current_timestep
        self._current_timestep = t

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=state.latents.device, dtype=state.latents.dtype)

        t_for_model = t.expand(state.latents.shape[0]).to(
            device=state.latents.device,
            dtype=state.latents.dtype,
        )

        noise_pred = self._predict_noise(
            state.latents,
            state.img_shapes,
            state.prompt_embeds,
            state.prompt_embeds_mask,
            t_for_model,
            self.attention_kwargs,
        )
        if state.do_true_cfg:
            negative_noise_pred = self._predict_noise(
                state.latents,
                state.img_shapes,
                state.negative_prompt_embeds,
                state.negative_prompt_embeds_mask,
                t_for_model,
                self.attention_kwargs,
            )
            combined_noise_pred = _combine_cfg_noise(
                noise_pred,
                negative_noise_pred,
                state.true_cfg_scale,
                True,
            )
            noise_pred = combined_noise_pred

        noise_pred = -noise_pred

        return noise_pred

    def step_scheduler(
        self,
        state: "DiffusionRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        if self.interrupt:
            return

        t = state.current_timestep
        state.latents = state.scheduler.step(
            noise_pred,
            t,
            state.latents,
            return_dict=False,
        )[0]
        state.step_index += 1

    def post_decode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        self._current_timestep = None

        height = state.sampling.height or self.default_sample_size * self.vae_scale_factor
        width = state.sampling.width or self.default_sample_size * self.vae_scale_factor
        output_type = kwargs.get("output_type", "pil")

        return self._decode_latents(state.latents, height, width, output_type)

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
        return_index: int | None = None,
    ) -> DiffusionOutput:
        extracted_prompt, negative_prompt = self._extract_prompts(req.prompts)
        prompt = extracted_prompt or prompt

        height = req.sampling_params.height or height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = (
            req.sampling_params.max_sequence_length or max_sequence_length or self.default_max_sequence_length
        )
        generator = req.sampling_params.generator or generator
        return_index = return_index if return_index is not None else self.default_return_index
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            max_sequence_length=max_sequence_length,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            latents=latents,
            attention_kwargs=attention_kwargs,
            return_index=return_index,
        )

        latents = ctx["latents"]
        timesteps = ctx["timesteps"]
        prompt_embeds = ctx["prompt_embeds"]
        prompt_embeds_mask = ctx["prompt_embeds_mask"]
        negative_prompt_embeds = ctx["negative_prompt_embeds"]
        negative_prompt_embeds_mask = ctx["negative_prompt_embeds_mask"]
        do_true_cfg = ctx["do_true_cfg"]
        true_cfg_scale = ctx["true_cfg_scale"]
        img_shapes = ctx["img_shapes"]

        self._num_timesteps = len(timesteps)

        for i, t in enumerate(timesteps):
            self._current_timestep = t
            if self.interrupt:
                continue

            t_for_model = t.expand(latents.shape[0]).to(
                device=latents.device,
                dtype=latents.dtype,
            )

            noise_pred = self._predict_noise(
                latents,
                img_shapes,
                prompt_embeds,
                prompt_embeds_mask,
                t_for_model,
                self.attention_kwargs,
            )
            if do_true_cfg:
                negative_noise_pred = self._predict_noise(
                    latents,
                    img_shapes,
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    t_for_model,
                    self.attention_kwargs,
                )
                combined_noise_pred = _combine_cfg_noise(
                    noise_pred,
                    negative_noise_pred,
                    true_cfg_scale,
                    True,
                )
                noise_pred = combined_noise_pred

            noise_pred = -noise_pred

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        self._current_timestep = None

        return self._decode_latents(latents, height, width, output_type)

    @property
    def _execution_device(self):
        return self.device

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.transformer.load_weights(weights)
