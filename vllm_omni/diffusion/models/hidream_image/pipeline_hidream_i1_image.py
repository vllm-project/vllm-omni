# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import math
import os
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    T5EncoderModel,
    T5Tokenizer,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hidream_image import HiDreamImageTransformer2DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.get_logger(__name__)


def get_hidream_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    if not os.path.exists(vae_config_path):
        raise FileNotFoundError(
            f"VAE config not found at {vae_config_path}. "
            "Please ensure the model path contains a valid VAE configuration."
        )
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
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
        timesteps (`torch.Tensor`): The timestep schedule from the scheduler.
        num_inference_steps (`int`): The number of inference steps.
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


class HiDreamI1ImagePipeline(nn.Module, CFGParallelMixin, DiffusionPipelineProfilerMixin, ProgressBarMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        llama_path = self.od_config.extras["auxiliary_text_encoder"]
        if llama_path is None:
            logger.warning(
                f"auxiliary_text_encoder is not provided. "
                f"Attempting to load default LLaMA model: {self.od_config.extras['default_llama_model_id']}"
            )
            llama_path = self.od_config.extras["default_llama_model_id"]
        logger.info(f"Loading LLAMA model from: {llama_path}")

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model, subfolder="text_encoder_2", local_files_only=local_files_only
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            model, subfolder="text_encoder_3", local_files_only=local_files_only
        )
        self.tokenizer_3 = T5Tokenizer.from_pretrained(
            model, subfolder="tokenizer_3", local_files_only=local_files_only
        )
        self.text_encoder_4 = LlamaForCausalLM.from_pretrained(llama_path, output_hidden_states=True, dtype=dtype).to(
            self.device
        )
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_path, use_fast=False)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, HiDreamImageTransformer2DModel)
        self.transformer = HiDreamImageTransformer2DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )

        self.stage = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # HiDreamImage latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.default_sample_size = 128
        if getattr(self, "tokenizer_4", None) is not None:
            self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        negative_prompt_4=None,
        prompt_embeds_t5=None,
        prompt_embeds_llama3=None,
        negative_prompt_embeds_t5=None,
        negative_prompt_embeds_llama3=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if prompt is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `pooled_prompt_embeds`: {pooled_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt_2 is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `pooled_prompt_embeds`: {pooled_prompt_embeds}."
                "Please make sure to only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds_t5 is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_3} and `prompt_embeds_t5`: {prompt_embeds_t5}."
                "Please make sure to only forward one of the two."
            )
        elif prompt_4 is not None and prompt_embeds_llama3 is not None:
            raise ValueError(
                f"Cannot forward both `prompt_4`: {prompt_4} and `prompt_embeds_llama3`: {prompt_embeds_llama3}."
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and pooled_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `pooled_prompt_embeds`."
                "Cannot leave both `prompt` and `pooled_prompt_embeds` undefined."
            )
        elif prompt is None and prompt_embeds_t5 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_t5`."
                "Cannot leave both `prompt` and `prompt_embeds_t5` undefined."
            )
        elif prompt is None and prompt_embeds_llama3 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_llama3`."
                "Cannot leave both `prompt` and `prompt_embeds_llama3` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")
        elif prompt_4 is not None and (not isinstance(prompt_4, str) and not isinstance(prompt_4, list)):
            raise ValueError(f"`prompt_4` has to be of type `str` or `list` but is {type(prompt_4)}")

        if negative_prompt is not None and negative_pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_pooled_prompt_embeds`:"
                f" {negative_pooled_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_pooled_prompt_embeds`:"
                f" {negative_pooled_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds_t5 is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds_t5`:"
                f" {negative_prompt_embeds_t5}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_4 is not None and negative_prompt_embeds_llama3 is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_4`: {negative_prompt_4} and `negative_prompt_embeds_llama3`:"
                f" {negative_prompt_embeds_llama3}. Please make sure to only forward one of the two."
            )

        if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is not None:
            if pooled_prompt_embeds.shape != negative_pooled_prompt_embeds.shape:
                raise ValueError(
                    "`pooled_prompt_embeds` and `negative_pooled_prompt_embeds`"
                    " must have the same shape when passed directly, but"
                    f" got: `pooled_prompt_embeds` {pooled_prompt_embeds.shape} != `negative_pooled_prompt_embeds`"
                    f" {negative_pooled_prompt_embeds.shape}."
                )
        if prompt_embeds_t5 is not None and negative_prompt_embeds_t5 is not None:
            if prompt_embeds_t5.shape != negative_prompt_embeds_t5.shape:
                raise ValueError(
                    "`prompt_embeds_t5` and `negative_prompt_embeds_t5`"
                    " must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_t5` {prompt_embeds_t5.shape} != `negative_prompt_embeds_t5`"
                    f" {negative_prompt_embeds_t5.shape}."
                )
        if prompt_embeds_llama3 is not None and negative_prompt_embeds_llama3 is not None:
            if prompt_embeds_llama3.shape != negative_prompt_embeds_llama3.shape:
                raise ValueError(
                    "`prompt_embeds_llama3` and `negative_prompt_embeds_llama3`"
                    " must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_llama3` {prompt_embeds_llama3.shape} != `negative_prompt_embeds_llama3`"
                    f" {negative_prompt_embeds_llama3.shape}."
                )

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        max_sequence_length: int = 128,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder_3.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_3.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(
                untruncated_ids[:, min(max_sequence_length, self.tokenizer_3.model_max_length) - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_3.model_max_length)} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(
            text_input_ids.to(self.device), attention_mask=attention_mask.to(self.device)
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        tokenizer,
        text_encoder,
        prompt: str | list[str],
        max_sequence_length: int = 128,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        model_max_length = self.tokenizer.model_max_length
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, model_max_length),
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {model_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)
        return prompt_embeds

    def _get_llama3_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        max_sequence_length: int = 128,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder_4.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_4.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_4.batch_decode(
                untruncated_ids[:, min(max_sequence_length, self.tokenizer_4.model_max_length) - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_4.model_max_length)} tokens: {removed_text}"
            )

        outputs = self.text_encoder_4(
            text_input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )

        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        prompt_3: str | list[str] | None = None,
        prompt_4: str | list[str] | None = None,
        dtype: torch.dtype | None = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        negative_prompt_3: str | list[str] | None = None,
        negative_prompt_4: str | list[str] | None = None,
        prompt_embeds_t5: list[torch.FloatTensor] | None = None,
        prompt_embeds_llama3: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds_t5: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds_llama3: list[torch.FloatTensor] | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 128,
        lora_scale: float | None = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = pooled_prompt_embeds.shape[0]

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, prompt, max_sequence_length, dtype
            )

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if len(negative_prompt) > 1 and len(negative_prompt) != batch_size:
                raise ValueError(f"negative_prompt must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, negative_prompt, max_sequence_length, dtype
            )

            if negative_pooled_prompt_embeds_1.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_1 = negative_pooled_prompt_embeds_1.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            if len(prompt_2) > 1 and len(prompt_2) != batch_size:
                raise ValueError(f"prompt_2 must be of length 1 or {batch_size}")

            pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, prompt_2, max_sequence_length, dtype
            )

            if pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                pooled_prompt_embeds_2 = pooled_prompt_embeds_2.repeat(batch_size, 1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

            if len(negative_prompt_2) > 1 and len(negative_prompt_2) != batch_size:
                raise ValueError(f"negative_prompt_2 must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, negative_prompt_2, max_sequence_length, dtype
            )

            if negative_pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_2 = negative_pooled_prompt_embeds_2.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1
            )

        if prompt_embeds_t5 is None:
            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            if len(prompt_3) > 1 and len(prompt_3) != batch_size:
                raise ValueError(f"prompt_3 must be of length 1 or {batch_size}")

            prompt_embeds_t5 = self._get_t5_prompt_embeds(prompt_3, max_sequence_length, dtype)

            if prompt_embeds_t5.shape[0] == 1 and batch_size > 1:
                prompt_embeds_t5 = prompt_embeds_t5.repeat(batch_size, 1, 1)

        if do_classifier_free_guidance and negative_prompt_embeds_t5 is None:
            negative_prompt_3 = negative_prompt_3 or negative_prompt
            negative_prompt_3 = [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3

            if len(negative_prompt_3) > 1 and len(negative_prompt_3) != batch_size:
                raise ValueError(f"negative_prompt_3 must be of length 1 or {batch_size}")

            negative_prompt_embeds_t5 = self._get_t5_prompt_embeds(negative_prompt_3, max_sequence_length, dtype)

            if negative_prompt_embeds_t5.shape[0] == 1 and batch_size > 1:
                negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(batch_size, 1, 1)

        if prompt_embeds_llama3 is None:
            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            if len(prompt_4) > 1 and len(prompt_4) != batch_size:
                raise ValueError(f"prompt_4 must be of length 1 or {batch_size}")

            prompt_embeds_llama3 = self._get_llama3_prompt_embeds(prompt_4, max_sequence_length, dtype)

            if prompt_embeds_llama3.shape[0] == 1 and batch_size > 1:
                prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, batch_size, 1, 1)

        if do_classifier_free_guidance and negative_prompt_embeds_llama3 is None:
            negative_prompt_4 = negative_prompt_4 or negative_prompt
            negative_prompt_4 = [negative_prompt_4] if isinstance(negative_prompt_4, str) else negative_prompt_4

            if len(negative_prompt_4) > 1 and len(negative_prompt_4) != batch_size:
                raise ValueError(f"negative_prompt_4 must be of length 1 or {batch_size}")

            negative_prompt_embeds_llama3 = self._get_llama3_prompt_embeds(
                negative_prompt_4, max_sequence_length, dtype
            )

            if negative_prompt_embeds_llama3.shape[0] == 1 and batch_size > 1:
                negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, batch_size, 1, 1)

        # duplicate pooled_prompt_embeds for each generation per prompt
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # duplicate t5_prompt_embeds for batch_size and num_images_per_prompt
        bs_embed, seq_len, _ = prompt_embeds_t5.shape
        if bs_embed == 1 and batch_size > 1:
            prompt_embeds_t5 = prompt_embeds_t5.repeat(batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate prompt_embeds_t5 of batch size {bs_embed}")
        prompt_embeds_t5 = prompt_embeds_t5.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_t5 = prompt_embeds_t5.view(batch_size * num_images_per_prompt, seq_len, -1)

        # duplicate llama3_prompt_embeds for batch_size and num_images_per_prompt
        _, bs_embed, seq_len, dim = prompt_embeds_llama3.shape
        if bs_embed == 1 and batch_size > 1:
            prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate prompt_embeds_llama3 of batch size {bs_embed}")
        prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, 1, num_images_per_prompt, 1)
        prompt_embeds_llama3 = prompt_embeds_llama3.view(-1, batch_size * num_images_per_prompt, seq_len, dim)

        if do_classifier_free_guidance:
            # duplicate negative_pooled_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len = negative_pooled_prompt_embeds.shape
            if bs_embed == 1 and batch_size > 1:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_pooled_prompt_embeds of batch size {bs_embed}")
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

            # duplicate negative_t5_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len, _ = negative_prompt_embeds_t5.shape
            if bs_embed == 1 and batch_size > 1:
                negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_prompt_embeds_t5 of batch size {bs_embed}")
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate negative_prompt_embeds_llama3 for batch_size and num_images_per_prompt
            _, bs_embed, seq_len, dim = negative_prompt_embeds_llama3.shape
            if bs_embed == 1 and batch_size > 1:
                negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_prompt_embeds_llama3 of batch size {bs_embed}")
            negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, 1, num_images_per_prompt, 1)
            negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.view(
                -1, batch_size * num_images_per_prompt, seq_len, dim
            )

        return (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = (
            f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this"
            " method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        )
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = (
            f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this"
            " method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        )
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = (
            f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}`"
            " is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        )
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = (
            f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is"
            " deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        )
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(self.device)
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
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

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

    def check_cfg_parallel_validity(self, true_cfg_scale: float, has_neg_prompt: bool):
        if get_classifier_free_guidance_world_size() == 1:
            return True

        if true_cfg_scale <= 1:
            logger.warning("CFG parallel is NOT working correctly when true_cfg_scale <= 1.")
            return False

        if not has_neg_prompt:
            logger.warning(
                "CFG parallel is NOT working correctly when there is no negative prompt or negative prompt embeddings."
            )
            return False
        return True

    def diffuse(
        self,
        prompt_embeds_t5: torch.Tensor,
        prompt_embeds_llama3: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
    ) -> torch.Tensor:
        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    encoder_hidden_states_t5=prompt_embeds_t5,
                    encoder_hidden_states_llama3=prompt_embeds_llama3,
                    pooled_embeds=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # TODO: Modify CFG guidance
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

                pbar.update()

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        prompt_3: str | list[str] | None = None,
        prompt_4: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        negative_prompt_3: str | list[str] | None = None,
        negative_prompt_4: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds_t5: torch.FloatTensor | None = None,
        prompt_embeds_llama3: torch.FloatTensor | None = None,
        negative_prompt_embeds_t5: torch.FloatTensor | None = None,
        negative_prompt_embeds_llama3: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 128,
        **kwargs,
    ):
        extracted_prompt, negative_prompt = self._extract_prompts(req.prompts)
        prompt = extracted_prompt or prompt

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor

        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        generator = req.sampling_params.generator or generator
        true_cfg_scale = req.sampling_params.true_cfg_scale or guidance_scale
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        prompt_embeds = kwargs.get("prompt_embeds", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)

        if prompt_embeds is not None:
            deprecation_message = (
                "The `prompt_embeds` argument is deprecated."
                " Please use `prompt_embeds_t5` and `prompt_embeds_llama3` instead."
            )
            deprecate("prompt_embeds", "0.35.0", deprecation_message)
            prompt_embeds_t5 = prompt_embeds[0]
            prompt_embeds_llama3 = prompt_embeds[1]

        if negative_prompt_embeds is not None:
            deprecation_message = (
                "The `negative_prompt_embeds` argument is deprecated."
                "Please use `negative_prompt_embeds_t5` and `negative_prompt_embeds_llama3` instead."
            )
            deprecate("negative_prompt_embeds", "0.35.0", deprecation_message)
            negative_prompt_embeds_t5 = negative_prompt_embeds[0]
            negative_prompt_embeds_llama3 = negative_prompt_embeds[1]

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(height * scale // division * division)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            prompt_embeds_t5=prompt_embeds_t5,
            prompt_embeds_llama3=prompt_embeds_llama3,
            negative_prompt_embeds_t5=negative_prompt_embeds_t5,
            negative_prompt_embeds_llama3=negative_prompt_embeds_llama3,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif pooled_prompt_embeds is not None:
            batch_size = pooled_prompt_embeds.shape[0]

        # TODO: CFG guidance configuration
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        # 3. Encode prompt
        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None
        (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds_t5=prompt_embeds_t5,
            prompt_embeds_llama3=prompt_embeds_llama3,
            negative_prompt_embeds_t5=negative_prompt_embeds_t5,
            negative_prompt_embeds_llama3=negative_prompt_embeds_llama3,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5], dim=0)
            prompt_embeds_llama3 = torch.cat([negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pooled_prompt_embeds.dtype,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        latents = self.diffuse(
            prompt_embeds_t5,
            prompt_embeds_llama3,
            pooled_prompt_embeds,
            latents,
            timesteps,
            do_true_cfg,
        )

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=image, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
