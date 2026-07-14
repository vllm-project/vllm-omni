# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import inspect
import json
import logging
import math
import os
from collections.abc import Iterable
from typing import ClassVar, cast

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import (
    StageBoundary,
    StagePayload,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import calculate_shift
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.models.step_mixin import DiffusionV2CFGStepMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_qwen_image_edit_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImageEditPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)
    latent_channels = vae_config.get("z_dim", 16)

    def pre_process_func(
        request: OmniDiffusionRequest,
    ):
        """Pre-process requests for QwenImageEditPipeline."""
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}

        # Only handles single image
        if not raw_image:  # None or empty list
            raise ValueError("""Received no input image. This model requires one input image to run.""")
        elif isinstance(raw_image, list):
            if len(raw_image) > 1:
                raise ValueError("""Received multiple input images. Only a single image is supported by this model.""")
            else:
                raw_image = raw_image[0]

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image)
        else:
            image = cast(PIL.Image.Image | torch.Tensor | np.ndarray, raw_image)

        image_size = image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = request.sampling_params.height or calculated_height
        width = request.sampling_params.width or calculated_width

        # Ensure dimensions are multiples of vae_scale_factor * 2
        height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)

        # Store calculated dimensions in request
        prompt["additional_information"]["calculated_height"] = calculated_height
        prompt["additional_information"]["calculated_width"] = calculated_width
        request.sampling_params.height = height
        request.sampling_params.width = width

        # Preprocess image
        if image is not None and not (
            isinstance(image, torch.Tensor) and len(image.shape) > 1 and image.shape[1] == latent_channels
        ):
            image = image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)

            # Store preprocessed image and prompt image in request
            prompt["additional_information"]["preprocessed_image"] = image
            prompt["additional_information"]["prompt_image"] = prompt_image
        request.prompt = prompt
        return request

    return pre_process_func


def get_qwen_image_edit_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-processing function for QwenImageEditPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_dimensions(target_area: float, ratio: float):
    """Calculate width and height from target area and aspect ratio."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


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


def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "argmax"
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist"):
        return (
            encoder_output.latent_dist.mode()
            if sample_mode == "argmax"
            else encoder_output.latent_dist.sample(generator)
        )
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class QwenImageEditPipeline(
    nn.Module,
    SupportImageInput,
    DiffusionV2CFGStepMixin,
    QwenImageCFGParallelMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    supports_step_execution: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    stage_payload_tensor_fields: ClassVar[dict[StageBoundary, tuple[str, ...]]] = {
        StageBoundary.ENCODE_TO_DIT: (
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
        ),
        StageBoundary.DIT_TO_DECODE: ("latents",),
    }
    stage_payload_scalar_fields: ClassVar[dict[StageBoundary, tuple[str, ...]]] = {
        StageBoundary.ENCODE_TO_DIT: (
            "do_true_cfg",
            "img_shapes",
            "txt_seq_lens",
            "negative_txt_seq_lens",
        ),
    }
    stage_payload_private_tensor_fields: ClassVar[dict[StageBoundary, tuple[str, ...]]] = {
        StageBoundary.ENCODE_TO_DIT: ("image_latents",),
    }
    stage_payload_private_scalar_fields: ClassVar[dict[StageBoundary, tuple[str, ...]]] = {
        StageBoundary.ENCODE_TO_DIT: (
            "cfg_normalize",
            "decode_height",
            "decode_width",
            "guidance_scale",
            "num_inference_steps",
            "output_type",
            "sigmas",
            "true_cfg_scale",
        ),
        StageBoundary.DIT_TO_DECODE: ("decode_height", "decode_width", "output_type"),
    }

    def pack_stage_state(
        self,
        state: DiffusionRequestState,
        boundary: StageBoundary,
    ) -> StagePayload:
        def state_fields(field_names: tuple[str, ...]) -> dict[str, object]:
            return {name: getattr(state, name) for name in field_names if getattr(state, name, None) is not None}

        def extra_fields(field_names: tuple[str, ...]) -> dict[str, object]:
            return {name: state.extra[name] for name in field_names if name in state.extra}

        return StagePayload(
            request_id=state.request_id,
            boundary=boundary,
            scalar_fields=state_fields(self.stage_payload_scalar_fields.get(boundary, ())),
            tensor_fields=state_fields(self.stage_payload_tensor_fields.get(boundary, ())),
            private_scalar_fields=extra_fields(self.stage_payload_private_scalar_fields.get(boundary, ())),
            private_tensor_fields=extra_fields(self.stage_payload_private_tensor_fields.get(boundary, ())),
        )

    def unpack_stage_state(
        self,
        payload: StagePayload,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        if payload.request_id != state.request_id:
            raise ValueError(
                f"StagePayload request_id {payload.request_id!r} does not match state {state.request_id!r}."
            )
        if payload.boundary not in (StageBoundary.ENCODE_TO_DIT, StageBoundary.DIT_TO_DECODE):
            raise ValueError(f"Unsupported stage payload boundary: {payload.boundary!r}.")
        for name, value in payload.scalar_fields.items():
            setattr(state, name, value)
        for name, value in payload.tensor_fields.items():
            setattr(state, name, value)
        state.extra.update(payload.private_scalar_fields)
        state.extra.update(payload.private_tensor_fields)
        if "cfg_normalize" in state.extra:
            state.sampling.cfg_normalize = bool(state.extra["cfg_normalize"])
        if "true_cfg_scale" in state.extra:
            state.sampling.true_cfg_scale = state.extra["true_cfg_scale"]
        if "output_type" in state.extra:
            state.sampling.output_type = state.extra["output_type"]
        if "sigmas" in state.extra:
            state.sampling.sigmas = state.extra["sigmas"]
        return state

    def _prepare_image_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        image = image.to(device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        else:
            image_latents = image
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        image_latent_height, image_latent_width = image_latents.shape[3:]
        return self._pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )

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
        self.device = get_local_device()
        model = od_config.model

        # Check if model is a local path
        local_files_only = os.path.isdir(model)

        # See pipeline_qwen_image_edit_plus: guard against transformers v5
        # multi-worker race on partial subfolder shard sets (Buildkite #1043).
        qwen_subfolders = ["scheduler", "text_encoder", "vae", "tokenizer", "processor"]
        prefetch_subfolders(
            model,
            qwen_subfolders,
            local_files_only=local_files_only,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        # ``from_pretrained_with_prefetch`` re-prefetches and retries on a
        # half-written cache (missing-shard ``OSError`` *and* the default
        # -config size-mismatch ``RuntimeError`` that ``retry_on_missing_shard``
        # could not recover) instead of crashing the worker.
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen2_5_VLForConditionalGeneration.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLQwenImage.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        ).to(self.device)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        self.transformer = QwenImageTransformer2DModel(
            od_config=od_config,
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.processor = from_pretrained_with_prefetch(
            Qwen2VLProcessor.from_pretrained,
            model,
            subfolder="processor",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        )

        self.stage = None

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
        self.tokenizer_max_length = 1024
        # Edit prompt template - different from generation template
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _check_generation_inputs(
        self,
        prompt,
        height,
        width,
        image=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

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
                f" {negative_prompt_embeds}. Make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `prompt_embeds_mask` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > self.tokenizer_max_length:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than {self.tokenizer_max_length} but is {max_sequence_length}"
            )

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        image: PIL.Image.Image | torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        """Get prompt embeddings with image support for editing."""
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # The edit template contains fixed multimodal scaffolding around the
        # instruction. Validate against the empty-template baseline so image
        # placeholder text does not consume the user's text budget.
        template_tokens = self.tokenizer(
            [template.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # Qwen-Image-Edit expands image placeholders into many vision tokens
        # inside the processor. `max_sequence_length` is meant to constrain the
        # prompt text length, so validate on the text template before image
        # token expansion.
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self.tokenizer_max_length,
            supported_max_sequence_length=self.tokenizer_max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen prompt template",
        )

        # Use processor to handle both text and image inputs
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        prompt_name: str = "prompt",
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt,
                image,
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

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

    def _edit_request_context(
        self,
        state: DiffusionRequestState,
    ) -> dict[str, object]:
        first_prompt = state.prompt
        if not isinstance(first_prompt, (str, dict)):
            raise TypeError("QwenImageEditPipeline expects a string or dict prompt.")
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")

        # Get preprocessed image from request (pre-processing is done in DiffusionEngine)
        if not isinstance(first_prompt, str) and "preprocessed_image" in (
            additional_information := first_prompt.get("additional_information", {})
        ):
            prompt_image = additional_information.get("prompt_image")
            image = additional_information.get("preprocessed_image")
            calculated_height = additional_information.get("calculated_height")
            calculated_width = additional_information.get("calculated_width")
            height = state.sampling.height
            width = state.sampling.width
        else:
            raise RuntimeError("Missing preprocess image that should have been created by the preprocess function.")
        if prompt_image is None:
            raise RuntimeError("Missing prompt_image that should have been created by the preprocess function.")
        if image is None:
            raise RuntimeError("Missing preprocessed_image that should have been created by the preprocess function.")
        if not torch.is_tensor(image):
            raise TypeError("QwenImageEditPipeline expects preprocessed_image to be a torch.Tensor.")
        if height is None or width is None:
            raise RuntimeError("QwenImageEditPipeline requires resolved height and width before prepare().")
        if calculated_height is None or calculated_width is None:
            raise RuntimeError("Missing calculated image size from Qwen-Image-Edit preprocess.")
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_image": prompt_image,
            "image": image,
            "height": int(height),
            "width": int(width),
            "calculated_height": int(calculated_height),
            "calculated_width": int(calculated_width),
        }

    def check_inputs(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        context = self._edit_request_context(state)
        self._check_generation_inputs(
            context["prompt"],
            context["height"],
            context["width"],
            context["image"],
            context["negative_prompt"],
            None,
            None,
            None,
            None,
            ["latents"],
            state.sampling.max_sequence_length or self.tokenizer_max_length,
        )
        if context["negative_prompt"] is None:
            logger.warning(
                "negative_prompt is not set. The official Qwen-Image-Edit model "
                "may produce lower-quality results without a negative_prompt. "
                "Qwen official repository recommends to use whitespace string as negative_prompt. "
                "Note: some distilled variants may not be affected by this."
            )
        return state

    def encode(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        context = self._edit_request_context(state)

        max_sequence_length = state.sampling.max_sequence_length or self.tokenizer_max_length
        true_cfg_scale = state.sampling.true_cfg_scale or 4.0
        if state.sampling.guidance_scale_provided:
            guidance_scale = state.sampling.guidance_scale
        else:
            guidance_scale = 1.0
        num_images_per_prompt = (
            state.sampling.num_outputs_per_prompt if state.sampling.num_outputs_per_prompt > 0 else 1
        )

        has_neg_prompt = context["negative_prompt"] is not None

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        state.prompt_embeds, state.prompt_embeds_mask = self.encode_prompt(
            prompt=context["prompt"],
            image=context["prompt_image"],  # Use resized image for prompt encoding
            prompt_embeds=None,
            prompt_embeds_mask=None,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if do_true_cfg:
            state.negative_prompt_embeds, state.negative_prompt_embeds_mask = self.encode_prompt(
                prompt=context["negative_prompt"],
                image=context["prompt_image"],  # Use same resized image for negative prompt encoding
                prompt_embeds=None,
                prompt_embeds_mask=None,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )
        else:
            state.negative_prompt_embeds = None
            state.negative_prompt_embeds_mask = None
        state.do_true_cfg = do_true_cfg
        batch_size = 1
        num_channels_latents = self.transformer.in_channels // 4
        state.extra["image_latents"] = self._prepare_image_latents(
            context["image"],
            batch_size * num_images_per_prompt,
            num_channels_latents,
            state.prompt_embeds.dtype,
            self.device,
            state.sampling.generator,
        )
        state.extra["cfg_normalize"] = True
        state.extra["decode_height"] = context["height"]
        state.extra["decode_width"] = context["width"]
        state.extra["guidance_scale"] = guidance_scale
        state.extra["num_inference_steps"] = state.sampling.num_inference_steps or 50
        state.extra["output_type"] = state.sampling.output_type or "pil"
        state.extra["sigmas"] = state.sampling.sigmas
        state.extra["true_cfg_scale"] = true_cfg_scale
        state.img_shapes = [
            [
                (1, context["height"] // self.vae_scale_factor // 2, context["width"] // self.vae_scale_factor // 2),
                (
                    1,
                    context["calculated_height"] // self.vae_scale_factor // 2,
                    context["calculated_width"] // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size
        state.txt_seq_lens = (
            state.prompt_embeds_mask.sum(dim=1).tolist() if state.prompt_embeds_mask is not None else None
        )
        state.negative_txt_seq_lens = (
            state.negative_prompt_embeds_mask.sum(dim=1).tolist()
            if state.negative_prompt_embeds_mask is not None
            else None
        )
        return state

    def prepare(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        if state.prompt_embeds is None:
            raise ValueError(f"Request {state.request_id} has no prompt_embeds after encode().")
        if state.img_shapes is None:
            raise ValueError(f"Request {state.request_id} has no img_shapes after encode/unpack.")
        if "image_latents" not in state.extra:
            raise ValueError(f"Request {state.request_id} has no image_latents after encode/unpack.")
        num_images_per_prompt = (
            state.sampling.num_outputs_per_prompt if state.sampling.num_outputs_per_prompt > 0 else 1
        )
        batch_size = 1
        num_channels_latents = self.transformer.in_channels // 4
        decode_height = state.extra.get("decode_height")
        decode_width = state.extra.get("decode_width")
        if decode_height is None or decode_width is None:
            raise ValueError(f"Request {state.request_id} has no decode size after encode/unpack.")

        state.latents, _ = self.prepare_latents(
            None,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            int(decode_height),
            int(decode_width),
            state.prompt_embeds.dtype,
            self.device,
            state.sampling.generator,
            state.sampling.latents,
        )
        timesteps, _ = self.prepare_timesteps(
            state.extra["num_inference_steps"],
            state.extra["sigmas"],
            state.latents.shape[1],
        )
        self._num_timesteps = len(timesteps)
        state.timesteps = timesteps

        # handle guidance
        guidance_scale = state.extra["guidance_scale"]
        self._guidance_scale = guidance_scale
        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            state.guidance = guidance.expand(state.latents.shape[0])
        else:
            state.guidance = None

        state.scheduler = self._new_request_scheduler()
        state.step_index = 0
        state.sampling.cfg_normalize = bool(state.extra["cfg_normalize"])
        return state

    def _build_denoise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list[int] | None,
        do_true_cfg: bool,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        negative_txt_seq_lens: list[int] | None,
        image_latents: torch.Tensor | None,
    ) -> tuple[dict[str, object], dict[str, object] | None, int]:
        timestep = timestep.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "return_dict": False,
            "attention_kwargs": self.attention_kwargs,
        }
        if do_true_cfg:
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": negative_txt_seq_lens,
                "return_dict": False,
                "attention_kwargs": self.attention_kwargs,
            }
        else:
            negative_kwargs = None
        return positive_kwargs, negative_kwargs, latents.size(1)

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str,
    ) -> torch.Tensor:
        if output_type == "latent":
            return latents
        self._current_timestep = None
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
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
        return self.vae.decode(latents, return_dict=False)[0][:, :, 0]

    def decode(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        self._current_timestep = None
        try:
            height = state.extra["decode_height"]
            width = state.extra["decode_width"]
            output_type = state.extra["output_type"]
        except KeyError as exc:
            raise ValueError(f"Request {state.request_id} is missing decode metadata.") from exc
        if state.latents is None:
            raise ValueError(f"Request {state.request_id} has no latents to decode.")
        image = self._decode_latents(state.latents, int(height), int(width), output_type)
        state.extra["decoded_output"] = DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )
        return state

    def postprocess(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionOutput:
        output = state.extra.get("decoded_output")
        if output is None:
            raise ValueError(f"Request {state.request_id} has not been decoded.")
        if not isinstance(output, DiffusionOutput):
            raise TypeError(f"Decoded output for request {state.request_id} must be a DiffusionOutput.")
        return output

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Forward pass for image editing."""
        if len(req.prompts) > 1:
            logger.warning(
                "This model only supports a single prompt, not a batched request. Taking only the first image for now."
            )
        state = DiffusionRequestState(
            request_id=req.request_id,
            sampling=req.sampling_params,
            prompt=req.prompts[0],
            kv_sender_info=req.kv_sender_info,
        )
        state = self.init_state(state)
        state = self.check_inputs(state)
        state = self.encode(state)
        state = self.prepare(state)
        state = self.diffuse(state)
        state = self.decode(state)
        return self.postprocess(state)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
