# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared recipe-driven runtime for LTX pipeline variants."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, ClassVar

import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size as get_guidance_parallel_world_size,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request
from vllm_omni.platforms import current_omni_platform

from . import ltx2_latents as latent_ops
from .ltx2_components import (
    LTXComponentProfile,
    detect_ltx_model_version,
    initialize_pipeline_components,
    resolve_ltx_component_profile,
)
from .ltx2_conditioning import LTXPromptContext, LTXTextConditioningMixin
from .ltx2_denoise import (
    LTXDenoiseContext,
    LTXForwardContext,
    LTXPhaseExecutor,
    LTXPhaseResult,
    build_transformer_kwargs,
    step_denoised_latents,
)
from .ltx2_guidance import (
    LTX_GUIDANCE_EXECUTOR,
    LTXGuidanceExecutor,
    LTXGuidancePlan,
)
from .ltx2_phase_adapter import LTXPhaseAdapterRuntime, build_ltx_phase_adapter
from .ltx2_recipes import (
    LTXPhaseRecipe,
    LTXPipelineRecipe,
    resolve_ltx_pipeline_recipe,
)
from .ltx2_request import (
    LTXRequestInputs,
    LTXRequestMixin,
    validate_pipeline_request,
)


def _expand_per_prompt_decode_value(
    value: float | list[float],
    *,
    prompt_batch_size: int,
    effective_batch_size: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list):
        return [value] * effective_batch_size
    if len(value) == 1:
        return value * effective_batch_size
    if len(value) == effective_batch_size:
        return value
    if prompt_batch_size > 0 and len(value) == prompt_batch_size and effective_batch_size % prompt_batch_size == 0:
        repeats = effective_batch_size // prompt_batch_size
        return [item for item in value for _ in range(repeats)]
    raise ValueError(
        f"`{field_name}` must have length 1, prompt batch size ({prompt_batch_size}), or effective batch size"
        f" ({effective_batch_size}); got {len(value)}."
    )


def _prepare_decode_timestep_conditioning(
    *,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    prompt_batch_size: int,
    effective_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_timestep_values = _expand_per_prompt_decode_value(
        decode_timestep,
        prompt_batch_size=prompt_batch_size,
        effective_batch_size=effective_batch_size,
        field_name="decode_timestep",
    )
    decode_noise_scale_values = (
        decode_timestep_values
        if decode_noise_scale is None
        else _expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )


class LTXRuntime(
    LTXRequestMixin,
    LTXTextConditioningMixin,
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Shared Omni runtime for recipe-driven LTX denoise phases."""

    pipeline_kind: ClassVar[str] = "one_stage"
    component_profile: ClassVar[LTXComponentProfile]
    pipeline_recipe: ClassVar[LTXPipelineRecipe]
    guidance_executor: ClassVar[LTXGuidanceExecutor] = LTX_GUIDANCE_EXECUTOR
    supports_request_batch = False
    connector_batches_cfg = False
    distributed_video_decode = True
    support_image_input = False
    dummy_run_num_frames = 1
    preserve_sp_padded_audio_duration = False
    reports_stage_durations = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        parallel_config = getattr(od_config, "parallel_config", None)
        if getattr(parallel_config, "ulysses_mode", "strict") == "advanced_uaa":
            raise ValueError(
                f"{self.__class__.__name__} does not support ulysses_mode='advanced_uaa'. "
                "Use the default ulysses_mode='strict' for LTX sequence parallelism."
            )
        self.model_version = detect_ltx_model_version(od_config.model, revision=getattr(od_config, "revision", None))
        self.component_profile = resolve_ltx_component_profile(self.pipeline_kind, self.model_version)
        self.pipeline_recipe = resolve_ltx_pipeline_recipe(self.pipeline_kind, self.model_version)
        if getattr(od_config, "cache_backend", "none") == "cache_dit" and not self.pipeline_recipe.supports_cache_dit:
            raise ValueError(
                f"{self.__class__.__name__} does not support cache_backend='cache_dit'. "
                "Cache-DiT is not qualified for this LTX recipe."
            )
        self._dit_modules = list(self.component_profile.dit_modules)
        self._encoder_modules = list(self.component_profile.encoder_modules)
        self._vae_modules = list(self.component_profile.vae_modules)
        self._resident_modules = list(self.component_profile.resident_modules)
        if self.model_version in ("2.3", "2.5"):
            self.preserve_sp_padded_audio_duration = True
            self.reports_stage_durations = True
        super().__init__()
        self._guidance_plan = LTXGuidancePlan.build(self.pipeline_recipe.request_guidance)
        initialize_pipeline_components(self, od_config)
        self._phase_adapter: LTXPhaseAdapterRuntime | None = build_ltx_phase_adapter(self)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _forward_request(
        self,
        req: DiffusionRequestBatch,
        *,
        image: Any | None = None,
        image_crf: int | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        stage_1_sigmas: list[float] | None = None,
        stage_2_sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
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
        max_sequence_length: int | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
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
            image_crf=image_crf,
        )
        image = self._resolve_request_image(req, image, request_inputs)
        if image is not None and not self.support_image_input:
            raise ValueError(f"{self.__class__.__name__} does not support `image` input.")
        request_sigmas = self._resolve_request_sigmas(req, sigmas)
        request_phase_sigmas = self._resolve_request_phase_sigmas(req, stage_1_sigmas, stage_2_sigmas)
        validate_pipeline_request(
            request_inputs,
            pipeline_recipe=self.pipeline_recipe,
            vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
            vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
            pipeline_name=self.__class__.__name__,
            request_sigmas=request_sigmas,
            request_phase_sigmas=request_phase_sigmas,
        )
        phase_adapter = getattr(self, "_phase_adapter", None)
        if phase_adapter is not None and any(
            getattr(sampling, "lora_request", None) is not None for sampling in req.sampling_params_list
        ):
            raise ValueError(
                f"{self.__class__.__name__} cannot compose a request LoRA with its internal phase adapter."
            )
        return self._run_recipe(
            req,
            request_inputs,
            request_sigmas=request_sigmas,
            request_phase_sigmas=request_phase_sigmas,
            image=image,
        )

    def _run_recipe(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        request_sigmas: list[float] | None,
        request_phase_sigmas: tuple[list[float] | None, ...] | None = None,
        image: Any | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Execute one- and multi-phase recipes through the same control flow."""
        phase_results: list[LTXPhaseResult] = []
        prompt_context = None
        for phase_index, phase_recipe in enumerate(self.pipeline_recipe.phases):
            override_sigmas = None if request_phase_sigmas is None else request_phase_sigmas[phase_index]
            phase_sigmas = (
                override_sigmas
                if override_sigmas is not None
                else (
                    request_sigmas
                    if request_sigmas is not None
                    else (list(phase_recipe.sigmas) if phase_recipe.sigmas is not None else None)
                )
            )
            self._enter_phase(phase_recipe)
            phase_inputs = self._build_phase_inputs(
                request_inputs,
                phase_recipe,
                phase_results[-1] if phase_results else None,
            )
            if phase_sigmas is not None:
                phase_inputs = replace(phase_inputs, num_inference_steps=len(phase_sigmas) - 1)
            noise_scale = phase_recipe.noise_scale
            if override_sigmas is not None and phase_recipe.input_transform == "spatial_upsample":
                noise_scale = float(override_sigmas[0])
            phase_result = self.run_phase(
                req,
                phase_inputs,
                noise_scale=noise_scale,
                sigmas=phase_sigmas,
                timesteps=None,
                attention_kwargs=None,
                phase_recipe=phase_recipe,
                image=image,
                prompt_context=prompt_context,
            )
            phase_results.append(phase_result)
            prompt_context = phase_result.forward_context.prompt_context

        final_context = phase_results[-1].forward_context
        output_phase = LTXPhaseResult(
            forward_context=final_context,
            video=phase_results[self.pipeline_recipe.video_output_phase].video,
            audio=phase_results[self.pipeline_recipe.audio_output_phase].audio,
        )
        return self.decode_phase(output_phase)

    def _enter_phase(self, phase: LTXPhaseRecipe) -> None:
        self._active_phase_name = phase.name
        self._guidance_plan = LTXGuidancePlan.build(phase.guidance)
        phase_adapter = getattr(self, "_phase_adapter", None)
        if phase_adapter is None:
            if phase.adapter_slot is not None:
                raise RuntimeError(f"LTX phase {phase.name!r} requires adapter slot {phase.adapter_slot!r}.")
            return
        phase_adapter.activate(phase.adapter_slot)

    def eval(self):
        result = super().eval()
        phase_adapter = getattr(self, "_phase_adapter", None)
        if phase_adapter is not None:
            phase_adapter.finalize()
        return result

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return latent_ops.prepare_video_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            dtype,
            device,
            generator,
            latents,
        )

    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        latents_normalized: bool = False,
    ) -> tuple[torch.Tensor, int, int]:
        return latent_ops.prepare_audio_latents(
            self,
            batch_size,
            num_channels_latents,
            audio_latent_length,
            num_mel_bins,
            noise_scale,
            dtype,
            device,
            generator,
            latents,
            latents_normalized,
        )

    @property
    def guidance_scale(self):
        return self._guidance_plan.spec.video.cfg_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_plan.spec.do_cfg

    @property
    def do_guidance(self):
        return len(self._guidance_plan.passes) > 1

    @property
    def interrupt(self):
        return self._interrupt

    def _transformer_cache_context(self, context_name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        if callable(cache_context):
            return cache_context(context_name)
        return nullcontext()

    def predict_noise(self, **kwargs):
        with self._transformer_cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = self.transformer(**kwargs)
        return noise_pred_video.float(), noise_pred_audio.float()

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale,
        cfg_normalize=False,
        kwargs: dict[str, Any] | None = None,
        **context: Any,
    ):
        if kwargs is not None:
            context = {**kwargs, **context}
        del cfg_normalize
        required = ("video_latents", "audio_latents", "video_sigma", "audio_sigma")
        if any(context.get(name) is None for name in required):
            raise ValueError("LTX x0-space CFG requires video/audio latents and sigmas.")
        video_pos, audio_pos = positive_noise_pred
        video_neg, audio_neg = negative_noise_pred
        return (
            self.guidance_executor.combine_cfg_velocity(
                context["video_latents"],
                video_pos,
                video_neg,
                context["video_sigma"],
                true_cfg_scale,
            ),
            self.guidance_executor.combine_cfg_velocity(
                context["audio_latents"],
                audio_pos,
                audio_neg,
                context["audio_sigma"],
                true_cfg_scale,
            ),
        )

    def _synchronize_guidance_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        guidance_parallel_ready: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (guidance_parallel_ready and get_guidance_parallel_world_size() > 1):
            return latents

        # CUDA async execution otherwise permits numerical drift to accumulate
        # across guidance-parallel denoise steps.
        latents = tuple(tensor.contiguous() for tensor in latents)
        current_omni_platform.synchronize()
        return latents

    def _setup_forward_runtime(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        attention_kwargs: dict[str, Any] | None,
    ) -> bool:
        self._guidance_plan = LTXGuidancePlan.build(request_inputs.guidance)
        del req, attention_kwargs
        self._interrupt = False
        guidance_world_size = get_guidance_parallel_world_size()
        self.guidance_executor.validate_guidance_world_size(self._guidance_plan, guidance_world_size)
        self.guidance_executor.warn_if_imbalanced(
            self._guidance_plan,
            guidance_world_size,
            getattr(self, "_active_phase_name", "generate"),
        )
        return self.do_guidance and guidance_world_size > 1

    def _check_forward_inputs(
        self,
        request_inputs: LTXRequestInputs,
        image: Any | None = None,
    ) -> None:
        self.check_inputs(
            prompt=request_inputs.prompt,
            height=request_inputs.height,
            width=request_inputs.width,
            prompt_embeds=request_inputs.prompt_embeds,
            negative_prompt_embeds=request_inputs.negative_prompt_embeds,
            prompt_attention_mask=request_inputs.prompt_attention_mask,
            negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
        )

    def _resolve_request_image(
        self,
        req: DiffusionRequestBatch,
        image: Any | None,
        request_inputs: LTXRequestInputs,
    ) -> Any | None:
        del req, request_inputs
        return image

    def _make_output(self, output: tuple[torch.Tensor, torch.Tensor]) -> DiffusionOutput:
        if self.reports_stage_durations:
            return DiffusionOutput(
                output=output,
                stage_durations=getattr(self, "stage_durations", None),
            )
        return DiffusionOutput(output=output)

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
            return self._make_output((latents, audio_latents))

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep_decode, decode_noise_scale_t = _prepare_decode_timestep_conditioning(
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                prompt_batch_size=prompt_batch_size,
                effective_batch_size=latents.shape[0],
                device=device,
                dtype=latents.dtype,
            )
            latents = (1 - decode_noise_scale_t) * latents + decode_noise_scale_t * noise

        dist_initialized = torch.distributed.is_initialized()
        is_output_rank = not dist_initialized or torch.distributed.get_rank() == 0
        vae_decode_needs_all_ranks = False
        is_distributed_vae_enabled = getattr(self.vae, "is_distributed_enabled", None)
        if self.distributed_video_decode and dist_initialized and callable(is_distributed_vae_enabled):
            # Distributed tiled decode is collective, so every rank must enter it.
            vae_decode_needs_all_ranks = bool(is_distributed_vae_enabled())

        should_decode_video = not self.distributed_video_decode or is_output_rank or vae_decode_needs_all_ranks
        if should_decode_video:
            video = self.vae.decode(latents.to(self.vae.dtype), timestep_decode, return_dict=False)[0]
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if self.distributed_video_decode and not is_output_rank:
            return self._make_output(
                (
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                )
            )

        if video.numel() > 0:
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        generated_mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        audio = self.vocoder(generated_mel)
        return self._make_output((video, audio))

    def _prepare_video_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: LTXPromptContext,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latents = self.prepare_latents(
            batch_size=prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            num_channels_latents=self.transformer.config.in_channels,
            height=request_inputs.height,
            width=request_inputs.width,
            num_frames=request_inputs.num_frames,
            noise_scale=noise_scale,
            dtype=prompt_context.positive_connector_prompt_embeds.dtype,
            device=device,
            generator=request_inputs.generator,
            latents=request_inputs.latents,
        )
        return latents, None

    def _resolve_video_latent_dimensions(self, request_inputs: LTXRequestInputs) -> tuple[int, int, int]:
        latent_num_frames, latent_height, latent_width = latent_ops.resolve_video_latent_shape(
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
            vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
        )
        latents = request_inputs.latents
        if latents is not None:
            if latents.ndim == 5:
                _, _, latent_num_frames, latent_height, latent_width = latents.shape
            elif latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, expected a packed 3D or unpacked 5D tensor."
                )
        return latent_num_frames, latent_height, latent_width

    def _prepare_audio_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: LTXPromptContext,
        *,
        device: torch.device,
        noise_scale: float,
    ) -> tuple[torch.Tensor, int, int, int]:
        duration_s = request_inputs.num_frames / request_inputs.frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        audio_num_frames = self._resolve_audio_latent_length(audio_num_frames, request_inputs.audio_latents)

        num_mel_bins = self.audio_vae.config.mel_bins if self.audio_vae is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels = self.audio_vae.config.latent_channels if self.audio_vae is not None else 8
        audio_latents, original_num_frames, padded_num_frames = self.prepare_audio_latents(
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            num_channels_latents=num_channels,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=prompt_context.positive_connector_audio_prompt_embeds.dtype,
            device=device,
            generator=request_inputs.generator,
            latents=request_inputs.audio_latents,
            latents_normalized=request_inputs.audio_latents_normalized,
        )
        return audio_latents, original_num_frames, padded_num_frames, latent_mel_bins

    def _resolve_audio_latent_length(
        self,
        requested_length: int,
        audio_latents: torch.Tensor | None,
    ) -> int:
        if audio_latents is None or audio_latents.ndim != 4:
            return requested_length

        provided_length = audio_latents.shape[2]
        if not self.preserve_sp_padded_audio_duration:
            return provided_length

        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_length = latent_ops.get_sp_padded_audio_latent_length(requested_length, int(sp_size))
        return requested_length if provided_length in {requested_length, padded_length} else provided_length

    def _prepare_denoise_context_for_guidance(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        return self.guidance_executor.prepare_denoise_context(
            self._guidance_plan,
            forward_ctx.guidance_parallel_ready,
            get_guidance_parallel_world_size(),
            denoise_ctx,
        )

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        del forward_ctx, denoise_ctx
        return self.guidance_executor.timestep_kwargs(
            ts,
            video_token_count,
            audio_token_count,
            expand_for_sequence_parallel=True,
        )

    def _video_guidance_model_sigma(
        self,
        sigma: torch.Tensor,
        denoise_ctx: LTXDenoiseContext,
    ) -> torch.Tensor:
        """Return the video timestep used to convert velocity predictions to x0."""
        del denoise_ctx
        return sigma

    def _build_transformer_kwargs(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        audio_encoder_attention_mask: torch.Tensor | None,
        ts: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_transformer_kwargs(
            self,
            forward_ctx,
            denoise_ctx,
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            ts=ts,
            attention_kwargs=attention_kwargs,
        )

    def _predict_noise_for_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: latent_ops.LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.guidance_executor.predict_noise(
            self,
            self._guidance_plan,
            index,
            timestep,
            state,
            forward_ctx,
            denoise_ctx,
            preserve_positive_velocity=forward_ctx.sampler == "euler_ancestral",
        )

    def _denoise_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: latent_ops.LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> latent_ops.LTXAVState:
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        noise_pred_video, noise_pred_audio = self._predict_noise_for_step(
            index,
            timestep,
            state,
            forward_ctx,
            denoise_ctx,
        )
        video, audio = step_denoised_latents(
            self,
            forward_ctx,
            denoise_ctx,
            noise_pred_video,
            noise_pred_audio,
            timestep,
        )
        audio = latent_ops.clear_audio_padding(audio, forward_ctx.original_audio_num_frames)
        return latent_ops.LTXAVState(video=video, audio=audio)

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = latent_ops.unpack_latents(
            latents,
            forward_ctx.latent_num_frames,
            forward_ctx.latent_height,
            forward_ctx.latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = latent_ops.denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        audio_latents = latent_ops.unpad_audio_latents(audio_latents, forward_ctx.original_audio_num_frames)
        audio_latents = latent_ops.denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = latent_ops.unpack_audio_latents(
            audio_latents,
            num_mel_bins=forward_ctx.latent_mel_bins,
        )
        return latents, audio_latents

    def run_phase(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        phase_recipe: LTXPhaseRecipe,
        image: Any | None = None,
        prompt_context: LTXPromptContext | None = None,
    ) -> LTXPhaseResult:
        """Prepare and execute one phase without decoding its output."""
        return LTXPhaseExecutor.run(
            self,
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            phase_recipe=phase_recipe,
            image=image,
            prompt_context=prompt_context,
        )

    def _build_phase_inputs(
        self,
        request_inputs: LTXRequestInputs,
        phase: LTXPhaseRecipe,
        previous_phase: LTXPhaseResult | None,
    ) -> LTXRequestInputs:
        """Resolve one phase from immutable request inputs and prior AV state."""
        divisor = phase.spatial_downscale
        if request_inputs.height % divisor != 0 or request_inputs.width % divisor != 0:
            raise ValueError(
                f"LTX phase {phase.name!r} cannot scale resolution "
                f"{request_inputs.width}x{request_inputs.height} by {divisor}."
            )
        height = request_inputs.height // divisor
        width = request_inputs.width // divisor
        latents = request_inputs.latents
        audio_latents = request_inputs.audio_latents
        audio_latents_normalized = request_inputs.audio_latents_normalized
        decode_timestep = request_inputs.decode_timestep
        decode_noise_scale = request_inputs.decode_noise_scale

        if phase.input_transform == "spatial_upsample":
            if previous_phase is None:
                raise ValueError(f"LTX phase {phase.name!r} requires a previous phase to upsample.")
            if previous_phase.video.ndim != 5:
                raise ValueError(f"LTX spatial upsampling expects a 5D video latent, got {previous_phase.video.shape}.")
            latents = self._spatial_upsample_phase(previous_phase.video)
            if previous_phase.audio_for_next_phase is not None:
                audio_latents = previous_phase.audio_for_next_phase
                audio_latents_normalized = True
            else:
                audio_latents = previous_phase.audio
                audio_latents_normalized = False
            decode_timestep = 0.0
            decode_noise_scale = None

            expected_shape = latent_ops.resolve_video_latent_shape(
                height,
                width,
                request_inputs.num_frames,
                vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
                vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
            )
            if latents.shape[2:] != expected_shape:
                raise ValueError(
                    f"LTX phase {phase.name!r} upsampler produced {tuple(latents.shape[2:])}, expected "
                    f"{expected_shape} for {width}x{height}."
                )
        elif phase.input_transform != "initial":
            raise ValueError(f"Unsupported LTX phase input transform: {phase.input_transform!r}.")

        guidance = request_inputs.guidance if phase.allow_guidance_override else phase.guidance
        return replace(
            request_inputs,
            height=height,
            width=width,
            num_inference_steps=phase.num_inference_steps or request_inputs.num_inference_steps,
            guidance=guidance,
            latents=latents,
            audio_latents=audio_latents,
            audio_latents_normalized=audio_latents_normalized,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
        )

    def _spatial_upsample_phase(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the recipe-managed spatial upsampler to unpacked latents."""
        latent_upsampler = getattr(self, "latent_upsampler", None)
        if latent_upsampler is None:
            raise RuntimeError("This LTX pipeline recipe requires a spatial latent upsampler.")
        dtype = getattr(latent_upsampler, "dtype", latents.dtype)
        return latent_upsampler(latents.to(device=self.device, dtype=dtype))

    def decode_phase(self, phase: LTXPhaseResult) -> DiffusionOutput | list[DiffusionOutput]:
        """Decode one completed phase and restore per-request outputs."""
        forward_ctx = phase.forward_context
        request_inputs = forward_ctx.request_inputs
        output = self._decode_output(
            latents=phase.video,
            audio_latents=phase.audio,
            output_type=request_inputs.output_type,
            connector_prompt_embeds=forward_ctx.prompt_context.connector_prompt_embeds,
            generator=request_inputs.generator,
            device=forward_ctx.device,
            decode_timestep=request_inputs.decode_timestep,
            decode_noise_scale=request_inputs.decode_noise_scale,
            prompt_batch_size=forward_ctx.batch_size,
        )
        if not self.supports_request_batch:
            return output
        return split_diffusion_output_by_request(
            output,
            forward_ctx.req,
            num_outputs_per_prompt=forward_ctx.num_videos_per_prompt,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
