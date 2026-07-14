# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, ClassVar

import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.lora.request import LoRARequest

from .ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    create_transformer_from_config,
    load_transformer_config,
)
from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
from .ltx2_latents import LTXAVState
from .ltx2_pipeline_base import (
    LTXPipelineBase,
    LTXPromptContext,
    LTXRequestInputs,
)
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

logger = init_logger(__name__)


def get_ltx2_post_process_func(
    od_config: OmniDiffusionConfig,
):
    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            return {"video": video, "audio": audio}
        return output

    return post_process_func


def _unwrap_request_tensor(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _get_prompt_field(prompt: Any, key: str) -> Any:
    if isinstance(prompt, str):
        return None
    value = prompt.get(key)
    if value is None:
        additional = prompt.get("additional_information")
        if isinstance(additional, dict):
            value = additional.get(key)
    return _unwrap_request_tensor(value)


class LTX2Pipeline(LTXPipelineBase):
    supports_request_batch = False

    component_profile = LTX2_COMPONENT_PROFILE
    one_stage_recipe = LTX2_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)

    # Audio is diffused jointly with video; warmup must size audio tokens.
    dummy_run_num_frames = 2

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

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # See ``hub_prefetch.py`` for the transformers v5 multi-worker subfolder
        # race; prefetch the whole component set before any from_pretrained.
        ltx2_subfolders = [
            "tokenizer",
            "text_encoder",
            "connectors",
            "vae",
            "audio_vae",
            "vocoder",
            "scheduler",
        ]
        prefetch_subfolders(model, ltx2_subfolders, local_files_only=local_files_only)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        # prefer mmap loading as default device is cuda, and the output of text encoder
        # could be deterministic.
        with torch.device("cpu"):
            self.text_encoder = from_pretrained_with_prefetch(
                Gemma3ForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=ltx2_subfolders,
                local_files_only=local_files_only,
                torch_dtype=dtype,
            ).to(self.device)
        self.connectors = from_pretrained_with_prefetch(
            LTX2TextConnectors.from_pretrained,
            model,
            subfolder="connectors",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLLTX2Video.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.audio_vae = from_pretrained_with_prefetch(
            AutoencoderKLLTX2Audio.from_pretrained,
            model,
            subfolder="audio_vae",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vocoder = from_pretrained_with_prefetch(
            LTX2Vocoder.from_pretrained,
            model,
            subfolder="vocoder",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        quant_config = getattr(self.od_config, "quantization_config", None)
        self.transformer = create_transformer_from_config(transformer_config, quant_config=quant_config)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer", None) is not None else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        tokenizer_max_length = 1024
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_max_length = self.tokenizer.model_max_length
            if tokenizer_max_length is None or tokenizer_max_length > 100000:
                encoder_config = getattr(self.text_encoder, "config", None)
                config_max_len = getattr(encoder_config, "max_position_embeddings", None)
                if config_max_len is None:
                    config_max_len = getattr(encoder_config, "max_seq_len", None)
                tokenizer_max_length = config_max_len or 1024
        self.tokenizer_max_length = int(tokenizer_max_length)

        self._guidance_scale = None
        self._guidance_rescale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        # `diffusers>=0.38` moved per_layer_masked_mean_norm inside the connector,
        # so pack to 3D here and let the connector handle normalization.
        prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

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

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when "
                    "passed directly, but got: `prompt_attention_mask` "
                    f"{prompt_attention_mask.shape} != `negative_prompt_attention_mask` "
                    f"{negative_prompt_attention_mask.shape}."
                )

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, true_cfg_scale, cfg_normalize=False):
        """Per-element CFG combine with guidance_rescale support."""
        (video_pos, audio_pos) = positive_noise_pred
        (video_neg, audio_neg) = negative_noise_pred
        video_combined = super().combine_cfg_noise(video_pos, video_neg, true_cfg_scale, cfg_normalize)
        audio_combined = super().combine_cfg_noise(audio_pos, audio_neg, true_cfg_scale, cfg_normalize)
        if self._guidance_rescale and self._guidance_rescale > 0:
            video_combined = rescale_noise_cfg(video_combined, video_pos, guidance_rescale=self._guidance_rescale)
            audio_combined = rescale_noise_cfg(audio_combined, audio_pos, guidance_rescale=self._guidance_rescale)
        return (video_combined, audio_combined)

    def _resolve_request_inputs(
        self,
        req: DiffusionRequestBatch,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        frame_rate: float | None,
        num_inference_steps: int | None,
        timesteps: list[int] | None,
        guidance_scale: float,
        guidance_rescale: float,
        num_videos_per_prompt: int | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        audio_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        output_type: str,
        max_sequence_length: int | None,
    ) -> LTXRequestInputs:
        sampling = req.sampling_params
        prompt = [item if isinstance(item, str) else (item.get("prompt") or "") for item in req.prompts] or prompt
        if all(isinstance(item, str) or item.get("negative_prompt") is None for item in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = [
                "" if isinstance(item, str) else (item.get("negative_prompt") or "") for item in req.prompts
            ]

        height = sampling.height or height or self.one_stage_recipe.height
        width = sampling.width or width or self.one_stage_recipe.width
        num_frames = sampling.num_frames or num_frames or self.one_stage_recipe.num_frames
        frame_rate = sampling.resolved_frame_rate or frame_rate or self.one_stage_recipe.frame_rate
        num_inference_steps = (
            sampling.num_inference_steps or num_inference_steps or self.one_stage_recipe.num_inference_steps
        )
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchEulerDiscreteScheduler.")

        num_videos_per_prompt = (
            sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else num_videos_per_prompt or 1
        )
        max_sequence_length = sampling.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        if sampling.guidance_scale_provided:
            guidance_scale = sampling.guidance_scale
        if sampling.guidance_rescale is not None:
            guidance_rescale = sampling.guidance_rescale

        if generator is None:
            generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        latents = sampling.latents if sampling.latents is not None else latents
        audio_latents = (
            sampling.audio_latents
            if sampling.audio_latents is not None
            else sampling.extra_args.get("audio_latents", audio_latents)
        )

        request_prompt_embeds = [_get_prompt_field(item, "prompt_embeds") for item in req.prompts]
        if any(value is not None for value in request_prompt_embeds):
            prompt_embeds = torch.stack(request_prompt_embeds)  # type: ignore[arg-type]
        request_negative_embeds = [_get_prompt_field(item, "negative_prompt_embeds") for item in req.prompts]
        if any(value is not None for value in request_negative_embeds):
            negative_prompt_embeds = torch.stack(request_negative_embeds)  # type: ignore[arg-type]

        request_prompt_masks = [
            _get_prompt_field(item, "prompt_attention_mask") or _get_prompt_field(item, "attention_mask")
            for item in req.prompts
        ]
        if any(value is not None for value in request_prompt_masks):
            prompt_attention_mask = torch.stack(request_prompt_masks)  # type: ignore[arg-type]
        request_negative_masks = [
            _get_prompt_field(item, "negative_prompt_attention_mask")
            or _get_prompt_field(item, "negative_attention_mask")
            for item in req.prompts
        ]
        if any(value is not None for value in request_negative_masks):
            negative_prompt_attention_mask = torch.stack(request_negative_masks)  # type: ignore[arg-type]

        if sampling.decode_timestep is not None:
            decode_timestep = sampling.decode_timestep
        if sampling.decode_noise_scale is not None:
            decode_noise_scale = sampling.decode_noise_scale
        if sampling.output_type is not None:
            output_type = sampling.output_type

        return LTXRequestInputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_videos_per_prompt=int(num_videos_per_prompt),
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=int(max_sequence_length),
        )

    def _prepare_prompt_context(
        self,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> LTXPromptContext:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
        )
        padding_side = getattr(self.tokenizer, "padding_side", "left")
        video_context, audio_context, attention_mask = self.connectors(
            prompt_embeds,
            prompt_attention_mask,
            padding_side=padding_side,
        )

        negative_video_context = None
        negative_audio_context = None
        negative_attention_mask = None
        if self.do_classifier_free_guidance:
            negative_video_context, negative_audio_context, negative_attention_mask = self.connectors(
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                padding_side=padding_side,
            )

        return LTXPromptContext(
            batch_size=batch_size,
            connector_prompt_embeds=video_context,
            connector_audio_prompt_embeds=audio_context,
            connector_attention_mask=attention_mask,
            positive_connector_prompt_embeds=video_context,
            positive_connector_audio_prompt_embeds=audio_context,
            positive_connector_attention_mask=attention_mask,
            negative_connector_prompt_embeds=negative_video_context,
            negative_connector_audio_prompt_embeds=negative_audio_context,
            negative_connector_attention_mask=negative_attention_mask,
        )

    def _denoise_step(
        self,
        _index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXAVState:
        prompt_context = forward_ctx.prompt_context
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        video_input = state.video.to(prompt_context.positive_connector_prompt_embeds.dtype)
        audio_input = state.audio.to(prompt_context.positive_connector_prompt_embeds.dtype)
        expanded_timestep = timestep.expand(video_input.shape[0])
        do_cfg = self.do_classifier_free_guidance
        positive_kwargs = self._build_transformer_kwargs(
            forward_ctx,
            denoise_ctx,
            hidden_states=video_input,
            audio_hidden_states=audio_input,
            encoder_hidden_states=prompt_context.positive_connector_prompt_embeds,
            audio_encoder_hidden_states=prompt_context.positive_connector_audio_prompt_embeds,
            encoder_attention_mask=prompt_context.positive_connector_attention_mask,
            audio_encoder_attention_mask=prompt_context.positive_connector_attention_mask,
            ts=expanded_timestep,
        )
        negative_kwargs = (
            {
                **positive_kwargs,
                "encoder_hidden_states": prompt_context.negative_connector_prompt_embeds,
                "audio_encoder_hidden_states": prompt_context.negative_connector_audio_prompt_embeds,
                "encoder_attention_mask": prompt_context.negative_connector_attention_mask,
                "audio_encoder_attention_mask": prompt_context.negative_connector_attention_mask,
            }
            if do_cfg
            else None
        )
        noise_pred_video, noise_pred_audio = self.predict_noise_maybe_with_cfg(
            do_true_cfg=do_cfg,
            true_cfg_scale=forward_ctx.request_inputs.guidance_scale,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )
        video, audio = self._step_denoised_latents(
            forward_ctx,
            denoise_ctx,
            noise_pred_video,
            noise_pred_audio,
            timestep,
        )
        return LTXAVState(video=video, audio=audio)

    def _decode_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        output_type: str,
        connector_prompt_embeds: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        prompt_batch_size: int,
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(output=(latents, audio_latents))

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )
            decode_timestep_values = decode_timestep
            if not isinstance(decode_timestep_values, list):
                decode_timestep_values = [decode_timestep_values] * prompt_batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep_values
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * prompt_batch_size

            timestep_decode = torch.tensor(
                decode_timestep_values,
                device=device,
                dtype=latents.dtype,
            )
            decode_noise_scale_tensor = torch.tensor(
                decode_noise_scale,
                device=device,
                dtype=latents.dtype,
            )[:, None, None, None, None]
            latents = (1 - decode_noise_scale_tensor) * latents + decode_noise_scale_tensor * noise

        video = self.vae.decode(latents.to(self.vae.dtype), timestep_decode, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        generated_mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        audio = self.vocoder(generated_mel)
        return DiffusionOutput(output=(video, audio))

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = LTX2_ONE_STAGE_RECIPE.guidance_scale,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> DiffusionOutput:
        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )
        return self._forward_impl(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
        )


class LTX2TwoStagesPipeline(nn.Module, SupportsComponentDiscovery):
    """LTX2TwoStagesPipeline is for two stages image to video generation"""

    dummy_run_num_frames = 2
    supports_request_batch = False

    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["pipe.vae", "pipe.audio_vae"]
    one_stage_pipeline_cls = LTX2Pipeline

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = od_config.model
        self.distilled = False
        # User provided model path may contain '/' in the end and basename function
        # will not return the expected directory name, so we need to remove it by normpath
        if "distilled" in os.path.basename(os.path.normpath(self.model_path)):
            self.distilled = True
        else:
            raise NotImplementedError(f"{self.model_path} is not supported for {self.__class__.__name__}.")

        self.pipe = self.one_stage_pipeline_cls(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae,
            od_config=od_config,
        )

        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.pipe,
            device=self.device,
            dtype=self.dtype,
            max_cached_adapters=od_config.max_cpu_loras,
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="pipe.transformer.",
                fall_back_to_pt=True,
            ),
        ]

    def _run_two_stage(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        image: Any | None = None,
    ) -> DiffusionOutput:
        stage1 = self.pipe._run_denoise_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )

        upscaled_video_latent = self.upsample_pipe(
            latents=stage1.video,
            output_type="latent",
            return_dict=False,
        )[0]

        if not self.distilled:
            lora_path = f"{self.model_path}/ltx-2-19b-distilled-lora-384.safetensors"
            lora_request = LoRARequest(
                lora_name="stage_2_distilled",
                lora_int_id=1,
                lora_path=lora_path,
            )
            self.lora_manager.set_active_adapter(lora_request, lora_scale=1.0)
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )

        stage2_inputs = replace(
            request_inputs,
            num_inference_steps=3,
            guidance_scale=1.0,
            latents=upscaled_video_latent,
            audio_latents=stage1.audio,
            output_type="np",
        )
        stage2 = self.pipe._run_denoise_phase(
            req,
            stage2_inputs,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            timesteps=None,
            attention_kwargs=attention_kwargs,
            prompt_context=stage1.forward_context.prompt_context,
        )
        return self.pipe._decode_and_split(stage2.forward_context, stage2.video, stage2.audio)

    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> DiffusionOutput:
        request_inputs = self.pipe._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )
        return self._run_two_stage(
            req,
            request_inputs,
            noise_scale=noise_scale,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX-2 T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
