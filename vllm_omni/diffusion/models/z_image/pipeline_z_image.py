# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import copy
import inspect
import json
import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import PIL.Image
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.utils import create_transformers_model
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    ZImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _broadcast_rows(tensor: torch.Tensor, num_rows: int, name: str) -> torch.Tensor:
    """Validate and broadcast a scalar or 1D tensor to ``num_rows`` rows."""
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D, got ndim={tensor.ndim}.")

    if tensor.shape[0] == num_rows:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(num_rows)
    raise ValueError(f"Expected 1 or {num_rows} values for {name}, got {tensor.shape[0]}.")


def get_post_process_func(
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

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(
        images: torch.Tensor,
    ):
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


# Copied from diffusers
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
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


class ZImagePipeline(nn.Module, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery):
    supports_request_batch = False
    supports_step_execution: ClassVar[bool] = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

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
                subfolder="text_encoder",
                revision=od_config.revision,
                prefix="text_encoder.",
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="vae",
                revision=od_config.revision,
                prefix="vae.",
            ),
        ]
        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        prefetch_subfolders(
            model,
            ["scheduler", "text_encoder", "vae", "tokenizer"],
            local_files_only=local_files_only,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        text_encoder_config = AutoConfig.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.text_encoder = create_transformers_model(
            AutoModelForCausalLM,
            od_config,
            hf_config=text_encoder_config,
        ).to(self._execution_device)
        if text_encoder_config.tie_word_embeddings:
            self.text_encoder.lm_head.weight = self.text_encoder.get_input_embeddings().weight

        vae_config = DistributedAutoencoderKL.load_config(model, subfolder="vae", local_files_only=local_files_only)
        self.vae = DistributedAutoencoderKL.from_config(vae_config).to(self._execution_device)
        self.transformer = ZImageTransformer2DModel(quant_config=od_config.quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        # Note: Context parallelism is applied centrally in registry.initialize_model()
        # following diffusers' pattern of enable_parallelism() at model loading time

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(prompt) != len(negative_prompt):
                raise ValueError(
                    "`prompt` and `negative_prompt` must have the same batch size, "
                    f"but got {len(prompt)} and {len(negative_prompt)}, respectively."
                )
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        max_sequence_length: int = 512,
    ) -> list[torch.FloatTensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if image is not None:
            if latents is not None:
                return latents.to(device=device, dtype=dtype)

            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != num_channels_latents:
                if isinstance(generator, list):
                    image_latents = [
                        retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                        for i in range(image.shape[0])
                    ]
                    image_latents = torch.cat(image_latents, dim=0)
                else:
                    image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

                image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                image_latents = image

            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )

            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
            return latents

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # Shared setup with the step-execution path lives in
        # ``_prepare_generation_context`` so request-mode and step-mode do not
        # drift.  ``forward`` owns only the per-step denoise loop + decode.
        ctx = self._prepare_generation_context(req.prompts, req.sampling_params)

        prompt_embeds = ctx["prompt_embeds"]
        negative_prompt_embeds = ctx["negative_prompt_embeds"]
        latents = ctx["latents"]
        timesteps = ctx["timesteps"]
        do_classifier_free_guidance = bool(ctx["do_classifier_free_guidance"])
        guidance_scale = ctx["guidance_scale"]
        cfg_normalization = ctx["cfg_normalization"]
        cfg_truncation = ctx["cfg_truncation"]
        output_type = ctx["output_type"]
        actual_batch_size = ctx["actual_batch_size"]
        device = self._execution_device

        callback_on_step_end: Callable[[int, int, dict], None] | None = None
        callback_on_step_end_tensor_inputs = ["latents"]

        # Precompute normalized timesteps once to avoid per-step GPU->CPU sync (.item() causes cudaStreamSynchronize)
        if isinstance(timesteps, torch.Tensor):
            timesteps_tensor = timesteps.to(device=device, dtype=torch.float32)
        else:
            timesteps_tensor = torch.as_tensor(timesteps, device=device, dtype=torch.float32)
        norm_timesteps = (1000 - timesteps_tensor) / 1000
        t_norm_list = norm_timesteps.cpu().tolist()
        if not isinstance(t_norm_list, list):
            t_norm_list = [t_norm_list]

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            # Normalized time for time-aware config (0 at start, 1 at end);
            # use precomputed to avoid .item() sync per step
            t_norm = t_norm_list[i]

            # Handle cfg truncation
            current_guidance_scale = guidance_scale
            if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                if t_norm > cfg_truncation:
                    current_guidance_scale = 0.0

            # Run CFG only if configured AND scale is non-zero
            apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0
            latents_typed = latents.to(self.od_config.dtype)

            if apply_cfg:
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents_typed
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            model_out_list = self.transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )[0]

            if apply_cfg:
                # Perform CFG
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]

                noise_pred = []
                for j in range(actual_batch_size):
                    pos = pos_out[j].float()
                    neg = neg_out[j].float()

                    pred = pos + current_guidance_scale * (pos - neg)

                    # Renormalization (torch.where avoids GPU->CPU sync from Python if/scalar comparison)
                    if cfg_normalization and float(cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(pred)
                        max_new_norm = ori_pos_norm * float(cfg_normalization)
                        scale = torch.where(
                            new_pos_norm > max_new_norm,
                            (max_new_norm / new_pos_norm.clamp(min=1e-12)).to(pred.dtype),
                            pred.new_tensor(1.0),
                        )
                        pred = pred * scale

                    noise_pred.append(pred)

                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
            if latents.dtype != torch.float32:
                raise ValueError(
                    "Z-Image scheduler must return FP32 latents to preserve the continuous-batching dtype invariant, "
                    f"but returned {latents.dtype}."
                )

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        return DiffusionOutput(output=image, stage_durations=stage_durations)

    def _prepare_generation_context(
        self,
        prompts,
        sampling,
    ) -> dict[str, Any]:
        """Prepare step-execution state with the same request setup as forward()."""
        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts]

        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]

        prompt_embeds = None
        negative_prompt_embeds = None

        image = None
        if prompts:
            if len(prompts) > 1:
                logger.warning(
                    "This model only supports a single prompt for img2img, not a batched request. "
                    "Taking only the first image for now."
                )
            first_prompt = prompts[0]
            if not isinstance(first_prompt, str):
                raw_image = first_prompt.get("multi_modal_data", {}).get("image")
                if raw_image is not None:
                    if isinstance(raw_image, list):
                        image = [PIL.Image.open(im) if isinstance(im, str) else raw_image[0] for im in raw_image[:1]]
                    else:
                        image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else raw_image

        explicit_strength = sampling.strength is not None
        strength = sampling.strength if explicit_strength else 0.6
        if explicit_strength and image is None:
            logger.warning(
                "strength parameter (%.2f) is only applicable for image-to-image (I2I) generation. "
                "It will be ignored for text-to-image (T2I) generation.",
                strength,
            )
            strength = None
        if image is not None and strength is not None and (strength < 0 or strength > 1):
            raise ValueError(f"The value of strength should be in [0.0, 1.0] but is {strength}")

        height = sampling.height or 1024
        width = sampling.width or 1024
        num_inference_steps = sampling.num_inference_steps or 50
        generator = sampling.generator
        sigmas = sampling.sigmas
        max_sequence_length = sampling.max_sequence_length or 512
        guidance_scale = sampling.guidance_scale
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        latents = sampling.latents

        cfg_normalization = sampling.cfg_normalize
        cfg_truncation = sampling.extra_args.get("cfg_truncation", 1.0)

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self._execution_device

        # NOTE: Intentionally NOT mutating pipeline-level ``self._*`` here.  In
        # step-mode this method runs once per admitted request, and every write
        # would race with the previous request under continuous batching.
        # ``forward`` (request-mode) reads these values from local variables
        # returned in the ctx dict below.
        do_classifier_free_guidance = guidance_scale > 0
        # 2. Define call parameters
        batch_size = len(prompt)

        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        # img2img mode: prepare latents from input image
        if image is not None:
            # Handle image list - take first image
            if isinstance(image, list):
                image = image[0]

            # Prepare image for VAE encoding using image_processor
            if not isinstance(image, torch.Tensor):
                init_image = self.image_processor.preprocess(image, height, width)
                image = init_image.to(dtype=torch.float32, device=device)

            # Initialize scheduler kwargs for img2img
            mu = calculate_shift(
                (height // self.vae_scale_factor // 2) * (width // self.vae_scale_factor // 2),
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            # First initialize timesteps in scheduler
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )

            # Then adjust timesteps based on strength
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

            if num_inference_steps < 1:
                raise ValueError(
                    f"After adjusting the num_inference_steps by strength parameter: "
                    f"{strength}, the number of pipeline steps is {num_inference_steps} "
                    f"which is < 1 and not appropriate for this pipeline."
                )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds[0].dtype,
                device,
                generator,
                latents,
                image,
                latent_timestep,
            )
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                torch.float32,
                device,
                generator,
                latents,
            )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        # 5. Prepare timesteps (T2I path only; I2I path resolved them above)
        if image is None:
            image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )

        self._num_timesteps = len(timesteps)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "latents": latents,
            "timesteps": timesteps,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "guidance_scale": guidance_scale,
            "cfg_normalization": cfg_normalization,
            "cfg_truncation": cfg_truncation,
            "output_type": sampling.output_type or "pil",
            "actual_batch_size": batch_size * num_images_per_prompt,
        }

    @staticmethod
    def _prompt_embeds_to_padded(
        prompt_embeds: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        seq_lens = [int(prompt_embed.shape[0]) for prompt_embed in prompt_embeds]
        padded = pad_sequence(prompt_embeds, batch_first=True, padding_value=0.0)
        mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
        for i, seq_len in enumerate(seq_lens):
            mask[i, :seq_len] = True
        return padded, mask, seq_lens

    @staticmethod
    def _padded_prompt_embeds_to_list(
        prompt_embeds: torch.Tensor | list[torch.Tensor],
        prompt_embeds_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(prompt_embeds, list):
            return prompt_embeds

        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds_mask is not None and prompt_embeds_mask.ndim == 1:
            prompt_embeds_mask = prompt_embeds_mask.unsqueeze(0)

        embeds_list = []
        for i in range(prompt_embeds.shape[0]):
            if prompt_embeds_mask is None:
                embeds_list.append(prompt_embeds[i])
            else:
                embeds_list.append(prompt_embeds[i][prompt_embeds_mask[i].bool()])
        return embeds_list

    @staticmethod
    def _extra_row_tensor(
        states,
        extra_key: str,
        *,
        device: torch.device,
        none_value: float,
    ) -> torch.Tensor:
        """Gather per-request scalar/vector metadata without a device sync.

        Used for per-row broadcast of scalars like ``cfg_normalization``.
        """
        values: list[torch.Tensor] = []
        for state in states:
            value = state.extra.get(extra_key)
            row_count = int(state.latents.shape[0])
            if value is None:
                value = none_value
            if isinstance(value, torch.Tensor):
                row_values = value.detach().to(device=device, dtype=torch.float32)
            else:
                row_values = torch.as_tensor(value, device=device, dtype=torch.float32)

            values.append(_broadcast_rows(row_values, row_count, extra_key))

        return torch.cat(values, dim=0)

    @staticmethod
    def _normalized_timestep_values(
        timesteps: torch.Tensor | list[torch.Tensor],
    ) -> list[float]:
        """Materialize CPU timestep metadata once during request admission."""
        if isinstance(timesteps, torch.Tensor):
            timestep_tensor = timesteps.detach().to(dtype=torch.float32).reshape(-1)
        else:
            timestep_tensor = torch.stack(
                [torch.as_tensor(timestep, dtype=torch.float32) for timestep in timesteps],
                dim=0,
            ).reshape(-1)
        values = ((1000 - timestep_tensor) / 1000).cpu().tolist()
        return [float(value) for value in values]

    @staticmethod
    def _cfg_active_rows(states) -> list[bool]:
        """Resolve per-row CFG activity entirely from admission-time CPU metadata."""
        active_rows: list[bool] = []
        for state in states:
            normalized_timesteps = state.extra.get("z_image_normalized_timesteps")
            if normalized_timesteps is None or state.step_index >= len(normalized_timesteps):
                raise ValueError(f"Missing normalized timestep metadata for Z-Image request {state.request_id}.")

            guidance_scale = float(state.extra.get("z_image_guidance_scale", 0.0))
            cfg_truncation = state.extra.get("z_image_cfg_truncation")
            if isinstance(cfg_truncation, torch.Tensor):
                raise ValueError("Z-Image CFG truncation must be stored as CPU metadata.")

            active = guidance_scale > 0
            if cfg_truncation is not None and float(cfg_truncation) <= 1:
                active = active and normalized_timesteps[state.step_index] <= float(cfg_truncation)
            active_rows.extend([active] * int(state.latents.shape[0]))
        return active_rows

    def prepare_encode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> "StepRequestState":
        del kwargs
        prompts = [state.prompt] if state.prompt is not None else []
        ctx = self._prepare_generation_context(prompts, state.sampling)

        req_scheduler = copy.deepcopy(self.scheduler)

        prompt_embeds, prompt_embeds_mask, txt_seq_lens = self._prompt_embeds_to_padded(ctx["prompt_embeds"])
        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.txt_seq_lens = txt_seq_lens

        if ctx["negative_prompt_embeds"]:
            negative_prompt_embeds, negative_prompt_embeds_mask, negative_txt_seq_lens = self._prompt_embeds_to_padded(
                ctx["negative_prompt_embeds"]
            )
            state.negative_prompt_embeds = negative_prompt_embeds
            state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
            state.negative_txt_seq_lens = negative_txt_seq_lens
        else:
            state.negative_prompt_embeds = None
            state.negative_prompt_embeds_mask = None
            state.negative_txt_seq_lens = None

        # Keep the persistent step state in the scheduler's FP32 domain. I2I
        # admission can otherwise produce BF16 latents while an already-running
        # I2I request has been promoted to FP32 by scheduler.step(), making a
        # later continuous-batching join fail with mixed latent dtypes.
        state.latents = ctx["latents"].to(torch.float32)
        state.timesteps = ctx["timesteps"]
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = bool(ctx["do_classifier_free_guidance"])
        state.guidance = torch.full(
            (state.latents.shape[0],),
            float(ctx["guidance_scale"]),
            dtype=state.latents.dtype,
            device=state.latents.device,
        )
        cfg_normalization = ctx["cfg_normalization"]
        cfg_normalization_row = (
            None
            if cfg_normalization is None
            else torch.full(
                (state.latents.shape[0],),
                float(cfg_normalization),
                dtype=torch.float32,
                device=state.latents.device,
            )
        )
        cfg_truncation = ctx["cfg_truncation"]
        state.extra["z_image_cfg_normalization"] = cfg_normalization_row
        state.extra["z_image_cfg_truncation"] = None if cfg_truncation is None else float(cfg_truncation)
        state.extra["z_image_guidance_scale"] = float(ctx["guidance_scale"])
        state.extra["z_image_normalized_timesteps"] = self._normalized_timestep_values(ctx["timesteps"])
        state.extra["z_image_output_type"] = ctx["output_type"]

        return state

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del kwargs

        prompt_embeds = self._padded_prompt_embeds_to_list(
            input_batch.prompt_embeds,
            input_batch.prompt_embeds_mask,
        )
        if input_batch.negative_prompt_embeds is None:
            negative_prompt_embeds = None
        else:
            negative_prompt_embeds = self._padded_prompt_embeds_to_list(
                input_batch.negative_prompt_embeds,
                input_batch.negative_prompt_embeds_mask,
            )

        actual_batch_size = int(input_batch.latents.shape[0])
        timestep = input_batch.timesteps.to(device=input_batch.latents.device, dtype=torch.float32)
        timestep = _broadcast_rows(timestep, actual_batch_size, "Z-Image timesteps")
        timestep = (1000 - timestep) / 1000
        active_cfg_rows = self._cfg_active_rows(input_batch.states) if input_batch.do_true_cfg else []
        apply_cfg = input_batch.do_true_cfg and any(active_cfg_rows)
        latents_typed = input_batch.latents.to(self.od_config.dtype)

        if apply_cfg:
            if negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds must be initialized when Z-Image CFG is active.")
            if input_batch.guidance is None:
                raise ValueError("guidance must be initialized when Z-Image CFG is active.")

            guidance_scales = torch.as_tensor(
                input_batch.guidance,
                dtype=torch.float32,
                device=input_batch.latents.device,
            )
            guidance_scales = _broadcast_rows(guidance_scales, actual_batch_size, "Z-Image guidance")

            cfg_normalizations = self._extra_row_tensor(
                input_batch.states,
                "z_image_cfg_normalization",
                device=input_batch.latents.device,
                none_value=0.0,
            )
            active_cfg_mask = torch.tensor(active_cfg_rows, dtype=torch.bool, device=input_batch.latents.device)
            guidance_scales = torch.where(
                active_cfg_mask,
                guidance_scales,
                torch.zeros_like(guidance_scales),
            )

            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
            timestep_model_input = timestep.repeat(2)
        else:
            latent_model_input = latents_typed
            prompt_embeds_model_input = prompt_embeds
            timestep_model_input = timestep

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
        )[0]

        if apply_cfg:
            pos_out = torch.stack([out.float() for out in model_out_list[:actual_batch_size]], dim=0)
            neg_out = torch.stack([out.float() for out in model_out_list[actual_batch_size:]], dim=0)

            row_shape = (actual_batch_size,) + (1,) * (pos_out.ndim - 1)
            noise_pred = pos_out + guidance_scales.view(row_shape) * (pos_out - neg_out)

            norm_dims = tuple(range(1, noise_pred.ndim))
            ori_pos_norm = torch.linalg.vector_norm(pos_out, dim=norm_dims)
            new_pos_norm = torch.linalg.vector_norm(noise_pred, dim=norm_dims)
            max_new_norm = ori_pos_norm * cfg_normalizations
            normalize_rows = (guidance_scales > 0) & (cfg_normalizations > 0) & (new_pos_norm > max_new_norm)
            normalization_scale = torch.where(
                normalize_rows,
                max_new_norm / new_pos_norm.clamp(min=1e-12),
                torch.ones_like(new_pos_norm),
            )
            noise_pred = noise_pred * normalization_scale.view(row_shape)
        else:
            noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        return -noise_pred

    def step_scheduler(
        self,
        state: "StepRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        del kwargs
        state.latents = state.scheduler.step(
            noise_pred.to(torch.float32), state.current_timestep, state.latents, return_dict=False
        )[0]
        if state.latents.dtype != torch.float32:
            raise ValueError(
                "Z-Image scheduler must return FP32 latents to preserve the continuous-batching join invariant for "
                f"request {state.request_id!r}, but returned {state.latents.dtype}."
            )
        state.step_index += 1

    def post_decode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        output_type = kwargs.get("output_type")
        if not output_type:
            output_type = state.extra.get("z_image_output_type") or "pil"
        if output_type == "latent":
            image = state.latents
        else:
            latents = state.latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        return DiffusionOutput(output=image, stage_durations=stage_durations)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        # Record components loaded by diffusers submodules to satisfy strict checks.
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        # downstream pipelines (e.g. MingImagePipeline) may set ``self.text_encoder = None`` when they
        # bring their own conditioning path.
        if self.text_encoder is not None:
            loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights
