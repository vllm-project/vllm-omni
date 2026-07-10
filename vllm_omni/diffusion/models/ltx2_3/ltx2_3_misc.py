# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 non-compute helpers.

This module holds request/config/weight-loading plumbing so the LTX-2.3
pipeline does not depend on LTX-2 pipeline internals for those utilities.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_3_guidance import _LTX23GuidanceParams
from .ltx2_3_recipes import LTX23PipelineRecipe
from .ltx2_3_transformer import LTX2VideoTransformer3DModel as LTX23VideoTransformer3DModel

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


@dataclass
class _LTX23RequestInputs:
    prompt: str | list[str] | None
    negative_prompt: str | list[str] | None
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    guidance_scale: float
    guidance_params: _LTX23GuidanceParams
    num_videos_per_prompt: int
    generator: torch.Generator | list[torch.Generator] | None
    latents: torch.Tensor | None
    audio_latents: torch.Tensor | None
    prompt_embeds: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    decode_timestep: float | list[float]
    decode_noise_scale: float | list[float] | None
    output_type: str
    max_sequence_length: int


def _get_audio_latents_from_sampling(sampling: Any) -> torch.Tensor | None:
    if sampling.audio_latents is not None:
        return sampling.audio_latents
    return sampling.extra_args.get("audio_latents")


class LTX23RequestMixin:
    _ltx23_recipe: LTX23PipelineRecipe

    @staticmethod
    def _get_extra_arg(extra_args: dict[str, Any], names: tuple[str, ...], default: Any) -> Any:
        for name in names:
            value = extra_args.get(name)
            if value is not None:
                return value
        return default

    @staticmethod
    def _normalize_stg_blocks(value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            if not value.strip():
                return ()
            value = [part.strip() for part in value.split(",")]
        elif isinstance(value, int):
            value = [value]
        return tuple(int(block) for block in value)

    def _resolve_guidance_params(
        self,
        sampling_params: Any,
        guidance_scale: float,
    ) -> _LTX23GuidanceParams:
        extra_args = sampling_params.extra_args or {}
        recipe = self._ltx23_recipe
        video_guidance = recipe.video_guidance
        audio_guidance = recipe.audio_guidance
        user_guidance_scale = bool(getattr(sampling_params, "guidance_scale_provided", False))
        audio_cfg_default = guidance_scale if user_guidance_scale else audio_guidance.cfg_scale
        guidance_rescale = getattr(sampling_params, "guidance_rescale", None)
        if guidance_rescale not in (None, 0.0):
            video_rescale_default = audio_rescale_default = float(guidance_rescale)
        else:
            video_rescale_default = video_guidance.rescale_scale
            audio_rescale_default = audio_guidance.rescale_scale
        return _LTX23GuidanceParams(
            video_cfg_scale=float(
                self._get_extra_arg(extra_args, ("video_cfg_scale", "video_cfg_guidance_scale"), guidance_scale)
            ),
            audio_cfg_scale=float(
                self._get_extra_arg(extra_args, ("audio_cfg_scale", "audio_cfg_guidance_scale"), audio_cfg_default)
            ),
            video_stg_scale=float(
                self._get_extra_arg(
                    extra_args,
                    ("video_stg_scale", "video_stg_guidance_scale"),
                    video_guidance.stg_scale,
                )
            ),
            audio_stg_scale=float(
                self._get_extra_arg(
                    extra_args,
                    ("audio_stg_scale", "audio_stg_guidance_scale"),
                    audio_guidance.stg_scale,
                )
            ),
            video_modality_scale=float(
                self._get_extra_arg(
                    extra_args,
                    ("video_modality_scale", "a2v_guidance_scale"),
                    video_guidance.modality_scale,
                )
            ),
            audio_modality_scale=float(
                self._get_extra_arg(
                    extra_args,
                    ("audio_modality_scale", "v2a_guidance_scale"),
                    audio_guidance.modality_scale,
                )
            ),
            video_rescale_scale=float(self._get_extra_arg(extra_args, ("video_rescale_scale",), video_rescale_default)),
            audio_rescale_scale=float(self._get_extra_arg(extra_args, ("audio_rescale_scale",), audio_rescale_default)),
            video_stg_blocks=self._normalize_stg_blocks(
                self._get_extra_arg(extra_args, ("video_stg_blocks",), video_guidance.stg_blocks)
            ),
            audio_stg_blocks=self._normalize_stg_blocks(
                self._get_extra_arg(extra_args, ("audio_stg_blocks",), audio_guidance.stg_blocks)
            ),
        )

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
        guidance_scale: float | None,
        num_videos_per_prompt: int | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        audio_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        decode_timestep: float | list[float] | None,
        decode_noise_scale: float | list[float] | None,
        output_type: str,
        max_sequence_length: int | None,
    ) -> _LTX23RequestInputs:
        sampling_params_list = req.sampling_params_list
        common_sampling_params = sampling_params_list[0]
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        recipe = self._ltx23_recipe
        height = common_sampling_params.height or height or recipe.height
        width = common_sampling_params.width or width or recipe.width
        num_frames = common_sampling_params.num_frames or num_frames or recipe.num_frames
        frame_rate = common_sampling_params.resolved_frame_rate or frame_rate or recipe.frame_rate
        num_inference_steps = (
            common_sampling_params.num_inference_steps or num_inference_steps or recipe.num_inference_steps
        )
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchEulerDiscreteScheduler.")

        num_videos_per_prompt = (
            common_sampling_params.num_outputs_per_prompt
            if common_sampling_params.num_outputs_per_prompt > 0
            else num_videos_per_prompt or 1
        )
        max_sequence_length = (
            common_sampling_params.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        )

        if common_sampling_params.guidance_scale_provided:
            guidance_scale = common_sampling_params.guidance_scale
        elif guidance_scale is None:
            guidance_scale = recipe.video_guidance.cfg_scale
        guidance_params = self._resolve_guidance_params(common_sampling_params, guidance_scale)
        for sampling_params in sampling_params_list[1:]:
            other_guidance_scale = (
                sampling_params.guidance_scale if sampling_params.guidance_scale_provided else guidance_scale
            )
            other_guidance_params = self._resolve_guidance_params(sampling_params, other_guidance_scale)
            if other_guidance_params != guidance_params:
                raise ValueError(
                    "LTX23Pipeline requires homogeneous guidance extra args within one request batch, but got "
                    f"{guidance_params} and {other_guidance_params}."
                )

        if generator is None:
            generator = req.collate_request_generators(num_videos_per_prompt, generator)

        latents = req.collate_request_tensors("latents", latents)
        audio_latents = DiffusionRequestBatch.collate_tensors(
            [_get_audio_latents_from_sampling(sampling) for sampling in sampling_params_list],
            "audio_latents",
            audio_latents,
        )

        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            req.prompts,
            {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_attention_mask": negative_prompt_attention_mask,
            },
            field_aliases={
                "prompt_attention_mask": ("prompt_attention_mask", "attention_mask"),
                "negative_prompt_attention_mask": (
                    "negative_prompt_attention_mask",
                    "negative_attention_mask",
                ),
            },
        )
        prompt_embeds = prompt_fields["prompt_embeds"]
        negative_prompt_embeds = prompt_fields["negative_prompt_embeds"]
        prompt_attention_mask = prompt_fields["prompt_attention_mask"]
        negative_prompt_attention_mask = prompt_fields["negative_prompt_attention_mask"]
        if prompt_embeds is not None:
            prompt = None
        if negative_prompt_embeds is not None:
            negative_prompt = None

        if common_sampling_params.decode_timestep is not None:
            decode_timestep = common_sampling_params.decode_timestep
        elif decode_timestep is None:
            decode_timestep = recipe.decode_timestep
        if common_sampling_params.decode_noise_scale is not None:
            decode_noise_scale = common_sampling_params.decode_noise_scale
        elif decode_noise_scale is None:
            decode_noise_scale = recipe.decode_noise_scale
        if common_sampling_params.output_type is not None:
            output_type = common_sampling_params.output_type

        return _LTX23RequestInputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            guidance_params=guidance_params,
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


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def detect_vocoder_output_sample_rate(model: str) -> int | None:
    """Detect the vocoder output sample rate from vocoder/config.json."""
    vocoder_config_path = os.path.join(model, "vocoder", "config.json")
    if not os.path.exists(vocoder_config_path):
        try:
            from huggingface_hub import hf_hub_download

            vocoder_config_path = hf_hub_download(model, "vocoder/config.json")
        except Exception:
            return None
    try:
        with open(vocoder_config_path) as f:
            cfg = json.load(f)
        return cfg.get("output_sampling_rate")
    except Exception:
        return None


def create_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
) -> LTX23VideoTransformer3DModel:
    """Create an LTX-2.3 transformer from a config dict."""
    if not config and quant_config is None:
        return LTX23VideoTransformer3DModel()

    signature = inspect.signature(LTX23VideoTransformer3DModel.__init__)
    allowed_keys = set(signature.parameters.keys())
    kwargs = {k: v for k, v in config.items() if k in allowed_keys}
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return LTX23VideoTransformer3DModel(**kwargs)


def load_ltx23_weights(module: torch.nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(module)
    return loader.load_weights(weights)
