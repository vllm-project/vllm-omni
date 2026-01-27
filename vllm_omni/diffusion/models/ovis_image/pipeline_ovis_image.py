# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Alibaba Ovis-Image Team and The HuggingFace Team. All rights reserved.
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
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2TokenizerFast, Qwen3Model
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import OvisImageTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def get_ovis_image_post_process_func(
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
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
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
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timetemps
    from the scheduler after the call. Handles custom timeteps. Any kwargs will be supplied to `scheduler.set_timeteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`, *optional*):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"the current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_timesteps = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"the current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OvisImagePipeline(
    nn.Module,
):
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
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        self.text_encoder = Qwen3Model.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )

        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self._execution_device
        )

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )

        self.transformer = OvisImageTransformer2DModel(od_config=od_config)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        self.tokenizer_max_length = 1024
        self.system_prompt = """Describe the image by detailing the color, quantity, text, shape, size, texture, spatial
        relationships of the objects and background: """
        self.user_prompt_begin_id = 28
        self.tokenizer_max_length = 256 + self.user_prompt_begin_id
        self.default_sample_size = 128

    def _get_messages(
        self,
        prompt: str | list[str] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        messages = []

        for each_prompt in prompt:
            message = [
                {
                    "role": "user",
                    "content": self.system_prompt + each_prompt,
                }
            ]
            message = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            messages.append(message)

        return messages

    def _get_ovis_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        messages = self._get_messages(prompt)

        batch_size = len(messages)

        tokens = self.tokenizer(
            messages,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = tokens.input_ids.to(device=device)
        attention_mask = tokens.attention_mask.to(device=device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        prompt_embeds = outputs.last_hidden_state
        prompt_embeds = prompt_embeds * attention_mask[..., None]
        prompt_embeds = prompt_embeds[:, self.user_prompt_begin_id :, :]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch.device
            num_images_per_prompt: (`int`):
                number of images that should be generated per prompt
            prompt_embeds: (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, text embeddings will be generated from `prompt` input argument.
        """

        device = device or self._execution_device

        if prompt_embeds is None:
            prompt_embeds = self._get_ovis_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        text_ids[..., 1] = text_ids[..., 1] + torch.arange(prompt_embeds.shape[1])[None, :]
        text_ids[..., 2] = text_ids[..., 2] + torch.arange(prompt_embeds.shape[1])[None, :]
        text_ids = text_ids.to(device=device, dtype=dtype)
        return prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"""`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are
                {height} and {width}. Dimension will be resized accordingly"""
            )

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"""`callback_on_step_end_tensor_inputs` has to contain the following keys:
        #         {self._callback_tensor_inputs.keys()}"""
        #     )

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
            raise ValueError(f"`prompt` has to be of type `str` or `list[str]` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: "
                f"{negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if max_sequence_length is not None and max_sequence_length > 256:
            raise ValueError(f"`max_sequence_length` has to be less than or equal to 256 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channel_latents, height, width):
        latents = latents.view(batch_size, num_channel_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channel_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2
        height = int(2 * (int(height) // (vae_scale_factor * 2)))
        width = int(2 * (int(width) // (vae_scale_factor * 2)))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channel_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = int(2 * (int(height) // (self.vae_scale_factor * 2)))
        width = int(2 * (int(width) // (self.vae_scale_factor * 2)))

        shape = (batch_size, num_channel_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channel_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

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
            self._execution_device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    def denoising(
        self,
        latents,
        timesteps,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_scale,
        do_classifier_free_guidance,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        text_ids,
        negative_text_ids,
        latent_image_ids,
    ):
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self._interrupt:
                break

            self._current_timestep = t
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            # Not used in this pipeline
            # if callback_on_step_end is not None:
            #     callback_kwargs = {}
            #     for k in callback_on_step_end_tensor_inputs:
            #         callback_kwargs[k] = locals()[k]
            #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            #     latents = callback_outputs.pop("latents", latents)
            #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        guidance_scale: float = 5.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 256,
    ) -> DiffusionOutput:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                not greater than `1`).
            guidance_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `guidance_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`list[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ovis_image.OvisImagePipelineOutput`] or `tuple`:
            [`~pipelines.ovis_image.OvisImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt
        height = req.height or self.default_sample_size * self.vae_scale_factor
        width = req.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.num_inference_steps or num_inference_steps
        guidance_scale = req.guidance_scale if req.guidance_scale is not None else guidance_scale
        generator = req.generator or generator
        req_num_outputs = getattr(req, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        # Steps:
        # 1. Check Inputs
        # 2. encode prompts
        # 4. Prepare latents
        # 5. Prepare timesteps
        # 6. diffusion latents
        # 7. decode latents
        # 8. post process outputs

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        device = self._execution_device
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )

        negative_text_ids = None
        if do_classifier_free_guidance:
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 4. Prepare latent variables
        num_channel_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channel_latents=num_channel_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 5. Prepare timesteps

        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, image_seq_len)

        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 6. Denoising loop

        # We set the index here to remove DtoH sync, helpful especially during compilation
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696

        latents = self.denoising(
            latents,
            timesteps,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale,
            do_classifier_free_guidance,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            text_ids,
            negative_text_ids,
            latent_image_ids,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
