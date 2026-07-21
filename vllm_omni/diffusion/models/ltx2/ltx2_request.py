# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request normalization shared by LTX pipeline variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


@dataclass
class LTXRequestInputs:
    """Resolved request values shared by all LTX execution variants."""

    prompt: str | list[str] | None
    negative_prompt: str | list[str] | None
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    guidance_scale: float
    guidance_rescale: float
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


def _unwrap_request_tensor(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _get_prompt_field(prompt: Any, *keys: str) -> Any:
    if isinstance(prompt, str):
        return None
    additional = prompt.get("additional_information")
    for key in keys:
        value = prompt.get(key)
        if value is None and isinstance(additional, dict):
            value = additional.get(key)
        if value is not None:
            return _unwrap_request_tensor(value)
    return None


def _get_audio_latents_from_sampling(sampling: Any) -> torch.Tensor | None:
    if sampling.audio_latents is not None:
        return sampling.audio_latents
    return sampling.extra_args.get("audio_latents")


class LTXRequestMixin:
    """Normalize serving requests without coupling them to a model version."""

    supports_request_batch = False
    supports_guidance_rescale = False

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        height: int,
        width: int,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
    ) -> None:
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 32 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")
        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got {prompt_embeds.shape} and {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when"
                    f" passed directly, but got {prompt_attention_mask.shape} and"
                    f" {negative_prompt_attention_mask.shape}."
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
        guidance_scale: float,
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
        guidance_rescale: float = 0.0,
    ) -> LTXRequestInputs:
        sampling_params_list = req.sampling_params_list
        sampling = sampling_params_list[0]
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
        if self.supports_guidance_rescale and sampling.guidance_rescale is not None:
            guidance_rescale = sampling.guidance_rescale

        if self.supports_request_batch:
            if generator is None:
                generator = req.collate_request_generators(num_videos_per_prompt, generator)
            latents = req.collate_request_tensors("latents", latents)
            audio_latents = DiffusionRequestBatch.collate_tensors(
                [_get_audio_latents_from_sampling(item) for item in sampling_params_list],
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
        else:
            if generator is None:
                generator = sampling.generator
            if generator is None and sampling.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
            latents = sampling.latents if sampling.latents is not None else latents
            sampling_audio_latents = _get_audio_latents_from_sampling(sampling)
            if sampling_audio_latents is not None:
                audio_latents = sampling_audio_latents

            request_prompt_embeds = [_get_prompt_field(item, "prompt_embeds") for item in req.prompts]
            if any(value is not None for value in request_prompt_embeds):
                prompt_embeds = torch.stack(request_prompt_embeds)  # type: ignore[arg-type]
            request_negative_embeds = [_get_prompt_field(item, "negative_prompt_embeds") for item in req.prompts]
            if any(value is not None for value in request_negative_embeds):
                negative_prompt_embeds = torch.stack(request_negative_embeds)  # type: ignore[arg-type]
            request_prompt_masks = [
                _get_prompt_field(item, "prompt_attention_mask", "attention_mask") for item in req.prompts
            ]
            if any(value is not None for value in request_prompt_masks):
                prompt_attention_mask = torch.stack(request_prompt_masks)  # type: ignore[arg-type]
            request_negative_masks = [
                _get_prompt_field(item, "negative_prompt_attention_mask", "negative_attention_mask")
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
