# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import logging
import math
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import (
    StageBoundary,
    StagePayload,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.models.qwen_image.rope_utils import txt_seq_lens_from_embeds
from vllm_omni.diffusion.models.step_mixin import DiffusionV2CFGStepMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_qwen_image_post_process_func(
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
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenImagePipeline(
    nn.Module,
    DiffusionV2CFGStepMixin,
    QwenImageCFGParallelMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    supports_request_batch = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    supports_step_execution: ClassVar[bool] = True
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
    stage_payload_private_tensor_fields: ClassVar[dict[StageBoundary, tuple[str, ...]]] = {}
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
        # Check if model is a local path
        local_files_only = os.path.isdir(model)

        # See pipeline_qwen_image_edit_plus: guard against transformers v5
        # multi-worker race on partial subfolder shard sets (Buildkite #1043).
        qwen_subfolders = ["scheduler", "text_encoder", "vae", "tokenizer"]
        prefetch_subfolders(
            model,
            qwen_subfolders,
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
        )
        # Qwen2.5-VL ships a vision tower that text-to-image does not use.
        # Drop it while the model is still on CPU, before moving to GPU, so
        # the vision tower never consumes GPU memory. Handle both transformers
        # layouts: newer puts visual under .model, older puts it directly on
        # the model.
        visual_owner = None
        if hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "visual"):
            visual_owner = self.text_encoder.model
        elif hasattr(self.text_encoder, "visual"):
            visual_owner = self.text_encoder
        if visual_owner is not None:
            del visual_owner.visual
        else:
            logger.warning("Qwen-Image: vision tower not found on text encoder; skipping drop")
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLQwenImage.from_pretrained,
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

        self.stage = None

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        # QwenImage latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor * 2
        # )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _check_generation_inputs(
        self,
        prompt,
        height,
        width,
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

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs},
        # but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
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
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
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
        # Validate only the user prompt contribution. The Qwen template also
        # adds a fixed suffix after the user text, so subtracting only
        # prompt_template_encode_start_idx would overcount near-limit prompts.
        template_tokens = self.tokenizer(
            [template.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self.tokenizer_max_length,
            supported_max_sequence_length=self.tokenizer_max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen prompt template",
        )
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
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
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

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
    ) -> torch.Tensor:
        # generator=torch.Generator(device="cuda").manual_seed(42)
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # image_seq_len = latents.shape[1]
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
        """Extract prompt and negative_prompt from OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    @staticmethod
    def _normalize_single_prompt_tensor(value: object | None, *, target_ndim: int) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise TypeError(f"Expected prompt tensor with ndim {target_ndim}, got {type(value)!r}.")
        if value.ndim == target_ndim - 1:
            return value.unsqueeze(0)
        if value.ndim == target_ndim:
            return value
        raise ValueError(f"Expected prompt tensor with ndim {target_ndim - 1} or {target_ndim}, got {value.ndim}.")

    def _state_generation_context(self, state: DiffusionRequestState) -> dict[str, Any]:
        sampling = state.sampling
        prompt, negative_prompt = self._extract_prompts([state.prompt] if state.prompt is not None else [])
        prompt_embeds = None
        prompt_embeds_mask = None
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if state.prompt is not None:
            prompt_embeds = DiffusionRequestBatch.get_prompt_field(state.prompt, "prompt_embeds")
            prompt_embeds_mask = DiffusionRequestBatch.get_prompt_field(state.prompt, "prompt_embeds_mask")
            negative_prompt_embeds = DiffusionRequestBatch.get_prompt_field(state.prompt, "negative_prompt_embeds")
            negative_prompt_embeds_mask = DiffusionRequestBatch.get_prompt_field(
                state.prompt,
                "negative_prompt_embeds_mask",
            )
        prompt_embeds = self._normalize_single_prompt_tensor(prompt_embeds, target_ndim=3)
        prompt_embeds_mask = self._normalize_single_prompt_tensor(prompt_embeds_mask, target_ndim=2)
        negative_prompt_embeds = self._normalize_single_prompt_tensor(negative_prompt_embeds, target_ndim=3)
        negative_prompt_embeds_mask = self._normalize_single_prompt_tensor(negative_prompt_embeds_mask, target_ndim=2)
        if prompt_embeds is not None:
            prompt = None
        if negative_prompt_embeds is not None:
            negative_prompt = None

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": sampling.num_inference_steps or 50,
            "sigmas": sampling.sigmas,
            "guidance_scale": sampling.guidance_scale if sampling.guidance_scale_provided else 1.0,
            "num_images_per_prompt": sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1,
            "generator": sampling.generator,
            "true_cfg_scale": sampling.true_cfg_scale or 4.0,
            "max_sequence_length": sampling.max_sequence_length or self.tokenizer_max_length,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "latents": sampling.latents,
            "attention_kwargs": {},
            "callback_on_step_end_tensor_inputs": ["latents"],
            "output_type": sampling.output_type or "pil",
        }

    def check_inputs(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        context = self._state_generation_context(state)
        self._check_generation_inputs(
            context["prompt"],
            context["height"],
            context["width"],
            context["negative_prompt"],
            context["prompt_embeds"],
            context["negative_prompt_embeds"],
            context["prompt_embeds_mask"],
            context["negative_prompt_embeds_mask"],
            context["callback_on_step_end_tensor_inputs"],
            context["max_sequence_length"],
        )
        return state

    def encode(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        context = self._state_generation_context(state)
        self._guidance_scale = context["guidance_scale"]
        self._attention_kwargs = context["attention_kwargs"]

        prompt = context["prompt"]
        prompt_embeds = context["prompt_embeds"]
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        has_neg_prompt = context["negative_prompt"] is not None or (
            context["negative_prompt_embeds"] is not None and context["negative_prompt_embeds_mask"] is not None
        )
        do_true_cfg = context["true_cfg_scale"] > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(context["true_cfg_scale"], has_neg_prompt)

        state.prompt_embeds, state.prompt_embeds_mask = self.encode_prompt(
            prompt=context["prompt"],
            prompt_embeds=context["prompt_embeds"],
            prompt_embeds_mask=context["prompt_embeds_mask"],
            num_images_per_prompt=context["num_images_per_prompt"],
            max_sequence_length=context["max_sequence_length"],
        )
        if do_true_cfg:
            state.negative_prompt_embeds, state.negative_prompt_embeds_mask = self.encode_prompt(
                prompt=context["negative_prompt"],
                prompt_embeds=context["negative_prompt_embeds"],
                prompt_embeds_mask=context["negative_prompt_embeds_mask"],
                num_images_per_prompt=context["num_images_per_prompt"],
                max_sequence_length=context["max_sequence_length"],
                prompt_name="negative_prompt",
            )
        else:
            state.negative_prompt_embeds = None
            state.negative_prompt_embeds_mask = None

        state.do_true_cfg = do_true_cfg
        state.img_shapes = [
            [
                (
                    1,
                    context["height"] // self.vae_scale_factor // 2,
                    context["width"] // self.vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        state.txt_seq_lens = txt_seq_lens_from_embeds(state.prompt_embeds)
        state.negative_txt_seq_lens = txt_seq_lens_from_embeds(state.negative_prompt_embeds)
        state.extra["cfg_normalize"] = True
        state.extra["decode_height"] = context["height"]
        state.extra["decode_width"] = context["width"]
        state.extra["guidance_scale"] = context["guidance_scale"]
        state.extra["num_inference_steps"] = context["num_inference_steps"]
        state.extra["output_type"] = context["output_type"]
        state.extra["sigmas"] = context["sigmas"]
        state.extra["true_cfg_scale"] = context["true_cfg_scale"]
        return state

    def prepare(
        self,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        if state.prompt_embeds is None:
            raise ValueError(f"Request {state.request_id} has no prompt_embeds after encode().")
        if state.img_shapes is None:
            raise ValueError(f"Request {state.request_id} has no img_shapes after encode/unpack.")
        decode_height = state.extra.get("decode_height")
        decode_width = state.extra.get("decode_width")
        if decode_height is None or decode_width is None:
            raise ValueError(f"Request {state.request_id} has no decode size after encode/unpack.")

        num_images_per_prompt = (
            state.sampling.num_outputs_per_prompt if state.sampling.num_outputs_per_prompt > 0 else 1
        )
        batch_size = max(state.prompt_embeds.shape[0] // num_images_per_prompt, 1)
        num_channels_latents = self.transformer.in_channels // 4
        state.latents = self.prepare_latents(
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
        image_latents: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
        """Build positive/negative kwargs and output_slice for one denoise step.

        Returns:
            (positive_kwargs, negative_kwargs, output_slice)
        """
        # Broadcast timestep to match batch size
        t_for_model = timestep.expand(latents.shape[0]).to(
            device=latents.device,
            dtype=latents.dtype,
        )

        # Concatenate image latents if available (editing pipelines)
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": t_for_model / 1000,
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
                "timestep": t_for_model / 1000,
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

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Unpack, normalize, and VAE-decode latents into a DiffusionOutput."""
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

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
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

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
        state.extra["decoded_output"] = self._decode_latents(
            state.latents,
            int(height),
            int(width),
            str(output_type),
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

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        states: list[DiffusionRequestState] = []
        for request in req.requests:
            state = DiffusionRequestState(
                request_id=request.request_id,
                sampling=request.sampling_params,
                prompt=request.prompt,
                kv_sender_info=request.kv_sender_info,
            )
            state = self.init_state(state)
            state = self.check_inputs(state)
            state = self.encode(state)
            state = self.prepare(state)
            states.append(state)

        cached_batch = None
        while True:
            active_states = [state for state in states if not state.denoise_completed]
            if not active_states:
                break
            if self.interrupt:
                break

            input_batch = self.build_step_batch(active_states, cached_batch=cached_batch)
            cached_batch = input_batch
            noise_pred = self.denoise_step(input_batch)
            if noise_pred is None:
                if self.interrupt:
                    break
                raise RuntimeError("denoise_step returned None without pipeline interrupt.")

            row_offset = 0
            for state in active_states:
                if state.latents is None:
                    raise ValueError(f"Request {state.request_id} has no latents after denoise_step().")
                next_row_offset = row_offset + int(state.latents.shape[0])
                self.step_scheduler(state, noise_pred[row_offset:next_row_offset])
                row_offset = next_row_offset
            if row_offset != int(noise_pred.shape[0]):
                raise ValueError(
                    f"Consumed {row_offset} noise rows, but denoise_step returned {int(noise_pred.shape[0])}."
                )

        outputs: list[DiffusionOutput] = []
        for state in states:
            state = self.decode(state)
            outputs.append(self.postprocess(state))
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class QwenImageDMD2Pipeline(DMD2PipelineMixin, QwenImagePipeline):
    """QwenImage pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
