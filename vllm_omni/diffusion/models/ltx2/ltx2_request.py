# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request normalization and validation shared by LTX pipeline variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_guidance import LTXGuidanceSpec, LTXModalityGuidance
from .ltx2_recipes import LTXPipelineRecipe


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
    guidance: LTXGuidanceSpec
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
    audio_latents_normalized: bool = False

    @property
    def guidance_scale(self) -> float:
        """Compatibility view for callers that only understand video CFG."""
        return self.guidance.video.cfg_scale


LTXCheckpointKind = Literal["distilled", "regular"]


def validate_ltx_checkpoint(
    scheduler_config: Mapping[str, Any],
    *,
    expected_kind: LTXCheckpointKind | None,
    pipeline_name: str,
) -> None:
    """Ensure scheduler metadata matches the checkpoint required by a profile."""
    if expected_kind is None:
        return

    if expected_kind == "distilled":
        if scheduler_config.get("use_dynamic_shifting", True) or scheduler_config.get("shift_terminal") is not None:
            raise ValueError(
                f"{pipeline_name} requires a merged distilled LTX2 checkpoint; "
                "the loaded scheduler metadata describes a non-distilled checkpoint."
            )
        return

    if not scheduler_config.get("use_dynamic_shifting", False) or scheduler_config.get("shift_terminal") is None:
        raise ValueError(
            f"{pipeline_name} requires a regular non-distilled LTX checkpoint; "
            "the loaded scheduler metadata describes a distilled checkpoint."
        )


def validate_pipeline_request(
    request_inputs: LTXRequestInputs,
    *,
    pipeline_recipe: LTXPipelineRecipe,
    vae_spatial_compression_ratio: int,
    vae_temporal_compression_ratio: int,
    pipeline_name: str,
    request_sigmas: list[float] | None = None,
) -> None:
    """Validate the request capabilities declared by an execution recipe."""
    alignment = vae_spatial_compression_ratio * pipeline_recipe.max_spatial_downscale
    if request_inputs.height % alignment != 0 or request_inputs.width % alignment != 0:
        raise ValueError(
            f"{pipeline_name} resolution must be divisible by {alignment}, got "
            f"{request_inputs.width}x{request_inputs.height}."
        )

    if request_inputs.num_frames < 1 or (request_inputs.num_frames - 1) % vae_temporal_compression_ratio != 0:
        raise ValueError(
            f"{pipeline_name} `num_frames` must be {vae_temporal_compression_ratio} * k + 1, "
            f"got {request_inputs.num_frames}."
        )

    if not pipeline_recipe.allow_request_latents and (
        request_inputs.latents is not None or request_inputs.audio_latents is not None
    ):
        raise ValueError(f"{pipeline_name} does not accept request-provided video or audio latents.")

    if not pipeline_recipe.allow_request_sigmas and request_sigmas is not None:
        raise ValueError(f"{pipeline_name} uses fixed phase sigma schedules.")

    if pipeline_recipe.fixed_num_inference_steps and (
        request_inputs.num_inference_steps != pipeline_recipe.num_inference_steps
    ):
        raise ValueError(
            f"{pipeline_name} uses {pipeline_recipe.num_inference_steps} fixed Stage 1 denoise steps; "
            f"got {request_inputs.num_inference_steps}."
        )

    negative_prompt = request_inputs.negative_prompt
    has_negative_prompt = (
        any(bool(value) for value in negative_prompt) if isinstance(negative_prompt, list) else bool(negative_prompt)
    )
    if not pipeline_recipe.allow_negative_prompt and (
        has_negative_prompt or request_inputs.negative_prompt_embeds is not None
    ):
        raise ValueError(f"{pipeline_name} uses positive-only guidance and does not accept negative conditioning.")

    first_phase = pipeline_recipe.phases[0]
    if not first_phase.allow_guidance_override and request_inputs.guidance != first_phase.guidance:
        raise ValueError(f"{pipeline_name} uses fixed positive-only guidance and cannot override it.")


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


def _get_extra_arg(sampling: Any, *names: str) -> Any:
    extra_args = sampling.extra_args or {}
    for name in names:
        value = extra_args.get(name)
        if value is not None:
            return value
    return None


def _normalize_stg_blocks(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"STG blocks must be an integer or a list of integers, got {value!r}.")
    return tuple(int(item) for item in value)


def _resolve_negative_prompts(
    prompts: list[Any],
    fallback: str | list[str],
) -> str | list[str]:
    if not prompts:
        return fallback
    if isinstance(fallback, list) and len(fallback) != len(prompts):
        raise ValueError(
            f"`negative_prompt` has batch size {len(fallback)}, but the request batch has size {len(prompts)}."
        )

    resolved = []
    for index, item in enumerate(prompts):
        request_value = None if isinstance(item, str) else item.get("negative_prompt")
        if request_value is not None:
            resolved.append(request_value)
        else:
            resolved.append(fallback[index] if isinstance(fallback, list) else fallback)
    return resolved


def _resolve_modality_guidance(
    sampling: Any,
    modality: str,
    default: LTXModalityGuidance,
) -> LTXModalityGuidance:
    cfg_aliases = (f"{modality}_cfg_scale", f"{modality}_cfg_guidance_scale")
    stg_aliases = (f"{modality}_stg_scale", f"{modality}_stg_guidance_scale")
    modality_aliases = (
        ("video_modality_scale", "a2v_guidance_scale")
        if modality == "video"
        else ("audio_modality_scale", "v2a_guidance_scale")
    )
    cfg_scale = _get_extra_arg(sampling, *cfg_aliases)
    stg_scale = _get_extra_arg(sampling, *stg_aliases)
    modality_scale = _get_extra_arg(sampling, *modality_aliases)
    rescale_scale = _get_extra_arg(sampling, f"{modality}_rescale_scale")
    stg_blocks = _get_extra_arg(sampling, f"{modality}_stg_blocks")
    return replace(
        default,
        cfg_scale=default.cfg_scale if cfg_scale is None else float(cfg_scale),
        stg_scale=default.stg_scale if stg_scale is None else float(stg_scale),
        modality_scale=default.modality_scale if modality_scale is None else float(modality_scale),
        rescale_scale=default.rescale_scale if rescale_scale is None else float(rescale_scale),
        stg_blocks=_normalize_stg_blocks(stg_blocks, default.stg_blocks),
    )


def _resolve_guidance_spec(
    sampling: Any,
    default: LTXGuidanceSpec,
    *,
    guidance_scale: float | None,
) -> LTXGuidanceSpec:
    video = _resolve_modality_guidance(sampling, "video", default.video)
    audio = _resolve_modality_guidance(sampling, "audio", default.audio)

    common_cfg = sampling.guidance_scale if sampling.guidance_scale_provided else guidance_scale
    if common_cfg is not None:
        video = replace(video, cfg_scale=float(common_cfg))
        audio = replace(audio, cfg_scale=float(common_cfg))

    return LTXGuidanceSpec(video=video, audio=audio)


class LTXRequestMixin:
    """Normalize serving requests without coupling them to a model version."""

    supports_request_batch = False

    @staticmethod
    def _validate_forward_options(
        *,
        timesteps: list[int] | None,
        guidance_rescale: float | None,
        noise_scale: float,
        return_dict: bool,
        attention_kwargs: dict[str, Any] | None,
    ) -> None:
        if timesteps is not None:
            raise ValueError("LTX does not support `timesteps`; use `sigmas` for a custom schedule.")
        if guidance_rescale not in (None, 0.0):
            raise ValueError(
                "LTX does not support the common `guidance_rescale`; use `video_rescale_scale` and "
                "`audio_rescale_scale`."
            )
        if noise_scale != 0.0:
            raise ValueError("LTX only supports `noise_scale=0.0`.")
        if return_dict is not True:
            raise ValueError("LTX only supports `return_dict=True`.")
        if attention_kwargs is not None:
            raise ValueError("LTX does not support public `attention_kwargs`.")

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        *,
        image: Any | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
        guidance_rescale: float | None = None,
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
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Validate and normalize the public LTX request surface."""
        self._validate_forward_options(
            timesteps=timesteps,
            guidance_rescale=guidance_rescale,
            noise_scale=noise_scale,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )
        if image is not None and not self.support_image_input:
            raise ValueError(f"{self.__class__.__name__} does not support `image` input.")

        return self._forward_request(
            req,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
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

    @staticmethod
    def _resolve_request_sigmas(
        req: DiffusionRequestBatch,
        fallback: list[float] | None,
    ) -> list[float] | None:
        request_sigmas = [sampling.sigmas for sampling in req.sampling_params_list]
        first = request_sigmas[0] if request_sigmas else None
        if any(item != first for item in request_sigmas[1:]):
            raise ValueError("Batched LTX requests must use identical custom sigmas.")
        return first if first is not None else fallback

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
        guidance_scale: float | None,
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
        sampling_params_list = req.sampling_params_list
        for sampling in sampling_params_list:
            if sampling.timesteps is not None:
                raise ValueError("LTX does not support `timesteps`; use `sigmas` for a custom schedule.")
            if sampling.guidance_rescale:
                raise ValueError(
                    "LTX does not support the common `guidance_rescale`; use `video_rescale_scale` and "
                    "`audio_rescale_scale`."
                )
            if _get_extra_arg(sampling, "flow_shift") is not None:
                raise ValueError("LTX does not support `flow_shift`; use `sigmas` for a custom schedule.")

        sampling = sampling_params_list[0]
        is_dummy_run = req.is_dummy_run()
        prompt = [item if isinstance(item, str) else (item.get("prompt") or "") for item in req.prompts] or prompt
        negative_prompt = _resolve_negative_prompts(
            req.prompts,
            negative_prompt if negative_prompt is not None else self.pipeline_recipe.negative_prompt,
        )

        height = sampling.height or height or self.pipeline_recipe.height
        width = sampling.width or width or self.pipeline_recipe.width
        num_frames = sampling.num_frames or num_frames or self.pipeline_recipe.num_frames
        frame_rate = sampling.resolved_frame_rate or frame_rate or self.pipeline_recipe.frame_rate
        num_inference_steps = (
            sampling.num_inference_steps or num_inference_steps or self.pipeline_recipe.num_inference_steps
        )
        if is_dummy_run and self.pipeline_recipe.fixed_num_inference_steps:
            num_inference_steps = self.pipeline_recipe.num_inference_steps
        num_inference_steps = max(int(num_inference_steps), 2)

        num_videos_per_prompt = (
            sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else num_videos_per_prompt or 1
        )
        max_sequence_length = sampling.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        guidance_specs = [
            _resolve_guidance_spec(
                item,
                self.pipeline_recipe.request_guidance,
                guidance_scale=guidance_scale,
            )
            for item in sampling_params_list
        ]
        guidance = guidance_specs[0]
        if is_dummy_run and not self.pipeline_recipe.phases[0].allow_guidance_override:
            guidance = self.pipeline_recipe.request_guidance
        if any(item != guidance for item in guidance_specs[1:]):
            raise ValueError("Batched LTX requests must use identical video/audio guidance parameters.")

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
            if prompt_embeds is not None:
                prompt = None
            if negative_prompt_embeds is not None:
                negative_prompt = None

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
            guidance=guidance,
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
