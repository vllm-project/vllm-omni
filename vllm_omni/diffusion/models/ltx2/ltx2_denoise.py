# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared denoise execution primitives for LTX pipelines."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_guidance import euler_step_from_velocity
from .ltx2_latents import LTXAVState, unpack_audio_latents, unpad_audio_latents

if TYPE_CHECKING:
    from .ltx2_conditioning import LTXPromptContext
    from .ltx2_recipes import LTXPhaseRecipe
    from .ltx2_request import LTXRequestInputs


@dataclass
class LTXForwardContext:
    """Immutable metadata and schedulers for one LTX denoise phase."""

    req: DiffusionRequestBatch
    request_inputs: LTXRequestInputs
    prompt_context: LTXPromptContext
    device: torch.device
    guidance_parallel_ready: bool
    attention_kwargs: dict[str, Any] | None
    latent_num_frames: int
    latent_height: int
    latent_width: int
    latent_mel_bins: int
    original_audio_num_frames: int
    padded_audio_num_frames: int
    timesteps: torch.Tensor
    sampler: str
    audio_scheduler: Any
    video_audio_step_adapter: Any

    @property
    def batch_size(self) -> int:
        return self.prompt_context.batch_size

    @property
    def num_videos_per_prompt(self) -> int:
        return self.request_inputs.num_videos_per_prompt


@dataclass
class LTXDenoiseContext:
    """Mutable AV state and positional metadata for a denoise phase."""

    latents: torch.Tensor
    audio_latents: torch.Tensor
    video_coords: torch.Tensor
    audio_coords: torch.Tensor
    audio_attention_mask: torch.Tensor | None = None
    conditioning_mask: torch.Tensor | None = None
    conditioning_mask_for_model: torch.Tensor | None = None


@dataclass
class LTXPhaseResult:
    """Denoised AV latents and the context used to produce them."""

    forward_context: LTXForwardContext
    video: torch.Tensor
    audio: torch.Tensor
    audio_for_next_phase: torch.Tensor | None = None


class LTXDenoisePipeline(Protocol):
    """Pipeline state required by :class:`LTXDenoiseExecutor`."""

    @property
    def interrupt(self) -> bool: ...

    def progress_bar(self, iterable=None, total=None): ...


LTXDenoiseStep = Callable[[int, torch.Tensor, LTXAVState], LTXAVState]

LTX_ANCESTRAL_ETA = 1.0
LTX_ANCESTRAL_S_NOISE = 1.0
LTX_ANCESTRAL_NOISE_SEED_OFFSET = 10000
LTX_OFFICIAL_DEFAULT_SEED = 10


class LTXDenoiseExecutor:
    """Run the one shared LTX denoise loop.

    Prediction and scheduler math remain injectable so structural refactors do
    not change the existing LTX2/LTX2.3 numerical paths. Guidance will replace
    that step policy independently.
    """

    @staticmethod
    def run(
        pipeline: LTXDenoisePipeline,
        state: LTXAVState,
        timesteps: Iterable[torch.Tensor],
        step: LTXDenoiseStep,
    ) -> LTXAVState:
        timesteps = tuple(timesteps)
        with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
            for index, timestep in enumerate(timesteps):
                if pipeline.interrupt:
                    continue
                state = step(index, timestep, state)
                progress_bar.update()
        return state


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


class LTXVideoAudioStepAdapter:
    """Expose the shared LTX Euler update through the distributed scheduler API."""

    def __init__(
        self,
        pipeline: Any,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        *,
        image_conditioned: bool,
        sampler: str = "euler",
        generator: torch.Generator | list[torch.Generator] | None = None,
        conditioning_mask: torch.Tensor | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._audio_scheduler = audio_scheduler
        self._latent_num_frames = latent_num_frames
        self._latent_height = latent_height
        self._latent_width = latent_width
        self._image_conditioned = image_conditioned
        self._sampler = sampler
        self._conditioning_mask = conditioning_mask
        self._ancestral_generators = (
            _make_ancestral_generators(generator, pipeline.device) if sampler == "euler_ancestral" else None
        )
        self._step_index = 0

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        del t, return_dict, generator
        if self._sampler == "euler_ancestral":
            video_out = _ancestral_euler_step_from_velocity(
                latents[0],
                noise_pred[0],
                self._pipeline.scheduler.sigmas,
                self._step_index,
                self._ancestral_generators,
            )
            if self._conditioning_mask is not None:
                mask = self._conditioning_mask.unsqueeze(-1).to(video_out.dtype)
                video_out = torch.lerp(video_out, latents[0], mask)
        elif self._image_conditioned:
            video_out = self._pipeline._step_video_latents_i2v(
                noise_pred[0],
                latents[0],
                self._step_index,
                self._latent_num_frames,
                self._latent_height,
                self._latent_width,
            )
        else:
            video_out = euler_step_from_velocity(
                latents[0],
                noise_pred[0],
                self._pipeline.scheduler.sigmas,
                self._step_index,
            )
        if self._sampler == "euler_ancestral":
            audio_out = _ancestral_euler_step_from_velocity(
                latents[1],
                noise_pred[1],
                self._audio_scheduler.sigmas,
                self._step_index,
                self._ancestral_generators,
            )
        else:
            audio_out = euler_step_from_velocity(
                latents[1],
                noise_pred[1],
                self._audio_scheduler.sigmas,
                self._step_index,
            )
        self._step_index += 1
        return ((video_out, audio_out),)


def _make_ancestral_generators(
    generator: torch.Generator | list[torch.Generator] | None,
    device: torch.device,
) -> torch.Generator | list[torch.Generator]:
    """Create the official independent ``seed + 10000`` noise stream."""

    def _offset(source: torch.Generator | None) -> torch.Generator:
        seed = LTX_OFFICIAL_DEFAULT_SEED if source is None else source.initial_seed()
        return torch.Generator(device=device).manual_seed(seed + LTX_ANCESTRAL_NOISE_SEED_OFFSET)

    if isinstance(generator, list):
        return [_offset(item) for item in generator]
    return _offset(generator)


def _randn_like_with_generators(
    sample: torch.Tensor,
    generators: torch.Generator | list[torch.Generator],
) -> torch.Tensor:
    if not isinstance(generators, list):
        return torch.randn(sample.shape, generator=generators, dtype=sample.dtype, device=sample.device)
    if len(generators) != sample.shape[0]:
        raise ValueError(
            f"LTX ancestral sampling received {len(generators)} generators for batch size {sample.shape[0]}."
        )
    return torch.cat(
        [
            torch.randn((1, *sample.shape[1:]), generator=item, dtype=sample.dtype, device=sample.device)
            for item in generators
        ]
    )


def _ancestral_euler_step_from_velocity(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    generators: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    """Apply the official LTX-2.5 rectified-flow ancestral Euler step."""
    sigma = sigmas[step_index].to(torch.float32)
    sigma_next = sigmas[step_index + 1].to(torch.float32)
    # Official X0Model materializes the denoised prediction in the model
    # dtype before the ancestral step promotes it back to float32.
    denoised_model_dtype = (sample.float() - velocity.float() * sigma).to(sample.dtype)
    denoised = denoised_model_dtype.float()
    if sigma_next == 0:
        return denoised.to(sample.dtype)
    if generators is None:
        raise ValueError("LTX ancestral Euler sampling requires a noise generator.")

    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * LTX_ANCESTRAL_ETA
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * sample.float() + (1.0 - sigma_down_ratio) * denoised

    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0).sqrt()
    noise = _randn_like_with_generators(sample, generators)
    x_next = (alpha_next / alpha_down) * x_next + noise.float() * LTX_ANCESTRAL_S_NOISE * renoise_coeff
    result = x_next.to(sample.dtype)
    return result


def _official_ltx_sigmas(
    scheduler: Any,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the official LTX one-stage sigma schedule in torch fp32."""
    config = scheduler.config
    base_anchor = config.get("base_image_seq_len", 1024)
    max_anchor = config.get("max_image_seq_len", 4096)
    base_shift = config.get("base_shift", 0.95)
    max_shift = config.get("max_shift", 2.05)

    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    slope = (max_shift - base_shift) / (max_anchor - base_anchor)
    # Official LTX one-stage pipelines omit the latent when constructing
    # this schedule, so the shift stays at the max sequence anchor.
    sigma_shift = max_anchor * slope + (base_shift - slope * base_anchor)
    exp_shift = math.exp(sigma_shift)
    sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)

    # Official non-distilled checkpoints explicitly set this to 0.1. Missing
    # or None means the loaded scheduler disabled terminal stretching.
    terminal = config.get("shift_terminal")
    if terminal is not None:
        non_zero = sigmas != 0
        one_minus_sigmas = 1.0 - sigmas[non_zero]
        scale = one_minus_sigmas[-1] / (1.0 - terminal)
        sigmas[non_zero] = 1.0 - one_minus_sigmas / scale
    return sigmas.to(dtype=torch.float32, device=device)


def _set_scheduler_sigmas(scheduler: Any, sigmas: torch.Tensor) -> torch.Tensor:
    sigmas = sigmas.to(torch.float32)
    timesteps = sigmas[:-1] * scheduler.config.get("num_train_timesteps", 1000)
    scheduler.sigmas = sigmas
    scheduler.timesteps = timesteps
    scheduler.num_inference_steps = len(timesteps)
    scheduler._step_index = None
    scheduler._begin_index = None
    return timesteps


def prepare_scheduler_stage(
    pipeline: Any,
    request_inputs: LTXRequestInputs,
    *,
    device: torch.device,
    sigmas: list[float] | None,
    timesteps: list[int] | None,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    use_official_sigma_schedule: bool,
    image_conditioned: bool = False,
    sampler: str = "euler",
    generator: torch.Generator | list[torch.Generator] | None = None,
    conditioning_mask: torch.Tensor | None = None,
) -> tuple[Any, Any, torch.Tensor]:
    if sigmas is not None and timesteps is not None:
        raise ValueError("Only one of `sigmas` or `timesteps` may be provided.")

    audio_scheduler = copy.deepcopy(pipeline.scheduler)
    video_audio_step_adapter = LTXVideoAudioStepAdapter(
        pipeline,
        audio_scheduler,
        latent_num_frames,
        latent_height,
        latent_width,
        image_conditioned=image_conditioned,
        sampler=sampler,
        generator=generator,
        conditioning_mask=conditioning_mask,
    )
    if sigmas is not None:
        scheduler_sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        if scheduler_sigmas.ndim != 1 or scheduler_sigmas.numel() < 2:
            raise ValueError("An LTX custom sigma schedule must contain at least two boundary values.")
        if scheduler_sigmas[-1] != 0:
            scheduler_sigmas = torch.cat([scheduler_sigmas, scheduler_sigmas.new_zeros(1)])
        timesteps_tensor = _set_scheduler_sigmas(pipeline.scheduler, scheduler_sigmas)
        _set_scheduler_sigmas(audio_scheduler, scheduler_sigmas.clone())
        return audio_scheduler, video_audio_step_adapter, timesteps_tensor

    if sigmas is None and timesteps is None and use_official_sigma_schedule:
        scheduler_sigmas = _official_ltx_sigmas(pipeline.scheduler, request_inputs.num_inference_steps, device)
        timesteps_tensor = _set_scheduler_sigmas(pipeline.scheduler, scheduler_sigmas)
        _set_scheduler_sigmas(audio_scheduler, scheduler_sigmas.clone())
        return audio_scheduler, video_audio_step_adapter, timesteps_tensor

    mu = calculate_shift(
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_image_seq_len", 1024),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.95),
        pipeline.scheduler.config.get("max_shift", 2.05),
    )
    retrieve_timesteps(
        audio_scheduler,
        request_inputs.num_inference_steps,
        device,
        timesteps,
        sigmas=sigmas,
        mu=mu,
    )
    timesteps_tensor, _ = retrieve_timesteps(
        pipeline.scheduler,
        request_inputs.num_inference_steps,
        device,
        timesteps,
        sigmas=sigmas,
        mu=mu,
    )
    return audio_scheduler, video_audio_step_adapter, timesteps_tensor


def prepare_rope_coords_stage(
    pipeline: Any,
    forward_ctx: LTXForwardContext,
    latents: torch.Tensor,
    audio_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    transformer = pipeline.transformer
    video_coords = transformer.rope.prepare_video_coords(
        latents.shape[0],
        forward_ctx.latent_num_frames,
        forward_ctx.latent_height,
        forward_ctx.latent_width,
        latents.device,
        fps=forward_ctx.request_inputs.frame_rate,
    )
    audio_coords = transformer.audio_rope.prepare_audio_coords(
        audio_latents.shape[0],
        forward_ctx.padded_audio_num_frames,
        audio_latents.device,
    )
    return video_coords, audio_coords


def _first_frame_keyframes_mask(reference: torch.Tensor, latent_num_frames: int) -> torch.Tensor:
    """Mark the first causal latent frame, matching official LTX-2.5."""
    if latent_num_frames <= 0:
        raise ValueError(f"LTX latent frame count must be positive, got {latent_num_frames}.")
    tokens_per_frame, remainder = divmod(reference.shape[1], latent_num_frames)
    if remainder:
        raise ValueError(
            f"LTX video token count {reference.shape[1]} is not divisible by {latent_num_frames} latent frames."
        )
    mask = reference.new_zeros((reference.shape[0], reference.shape[1], 1))
    mask[:, :tokens_per_frame] = 1
    return mask


def build_transformer_kwargs(
    pipeline: Any,
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
    del encoder_attention_mask, audio_encoder_attention_mask
    keyframes_mask = None
    if getattr(pipeline.transformer.config, "use_keyframes_abs_pos_embedding", False):
        keyframes_mask = _first_frame_keyframes_mask(hidden_states, forward_ctx.latent_num_frames)
    return {
        "hidden_states": hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "keyframes_mask": keyframes_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "audio_encoder_hidden_states": audio_encoder_hidden_states,
        **pipeline._denoise_timestep_kwargs(
            ts,
            forward_ctx,
            denoise_ctx,
            video_token_count=hidden_states.shape[1],
            audio_token_count=audio_hidden_states.shape[1],
        ),
        # This is valid only because LTX connectors replace every padding token
        # with a learned register, making all output context tokens valid.
        "encoder_attention_mask": None,
        "audio_encoder_attention_mask": None,
        "audio_attention_mask": denoise_ctx.audio_attention_mask,
        "num_frames": forward_ctx.latent_num_frames,
        "height": forward_ctx.latent_height,
        "width": forward_ctx.latent_width,
        "fps": forward_ctx.request_inputs.frame_rate,
        "audio_num_frames": forward_ctx.padded_audio_num_frames,
        "video_coords": denoise_ctx.video_coords,
        "audio_coords": denoise_ctx.audio_coords,
        "attention_kwargs": forward_ctx.attention_kwargs if attention_kwargs is None else attention_kwargs,
        "return_dict": False,
    }


def step_denoised_latents(
    pipeline: Any,
    forward_ctx: LTXForwardContext,
    denoise_ctx: LTXDenoiseContext,
    noise_pred_video: torch.Tensor,
    noise_pred_audio: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = pipeline.scheduler_step_maybe_with_cfg(
        (noise_pred_video, noise_pred_audio),
        (timestep, timestep),
        (denoise_ctx.latents, denoise_ctx.audio_latents),
        do_true_cfg=pipeline.do_classifier_free_guidance,
        per_request_scheduler=forward_ctx.video_audio_step_adapter,
    )
    return pipeline._synchronize_guidance_parallel_step_output(
        latents,
        guidance_parallel_ready=forward_ctx.guidance_parallel_ready,
    )


class LTXPhaseExecutor:
    """Prepare and execute one LTX phase without owning model modules."""

    @staticmethod
    def run(
        pipeline: Any,
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
        pipeline._check_forward_inputs(request_inputs, image=image)
        guidance_parallel_ready = pipeline._setup_forward_runtime(req, request_inputs, attention_kwargs)
        device = pipeline.device
        if prompt_context is None:
            prompt_context = pipeline._prepare_prompt_context(
                prompt=request_inputs.prompt,
                negative_prompt=request_inputs.negative_prompt,
                prompt_embeds=request_inputs.prompt_embeds,
                negative_prompt_embeds=request_inputs.negative_prompt_embeds,
                prompt_attention_mask=request_inputs.prompt_attention_mask,
                negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
                num_videos_per_prompt=request_inputs.num_videos_per_prompt,
                max_sequence_length=request_inputs.max_sequence_length,
            )

        latent_num_frames, latent_height, latent_width = pipeline._resolve_video_latent_dimensions(request_inputs)
        latents, conditioning_mask = pipeline._prepare_video_latents_stage(
            request_inputs,
            prompt_context,
            device=device,
            noise_scale=noise_scale,
            image=image,
        )
        audio_latents, original_audio_num_frames, padded_audio_num_frames, latent_mel_bins = (
            pipeline._prepare_audio_latents_stage(
                request_inputs,
                prompt_context,
                device=device,
                noise_scale=noise_scale,
            )
        )
        audio_scheduler, video_audio_step_adapter, timesteps_tensor = prepare_scheduler_stage(
            pipeline,
            request_inputs,
            device=device,
            sigmas=sigmas,
            timesteps=timesteps,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            use_official_sigma_schedule=phase_recipe.use_official_sigma_schedule,
            image_conditioned=conditioning_mask is not None,
            sampler=phase_recipe.sampler,
            generator=request_inputs.generator,
            conditioning_mask=conditioning_mask,
        )
        forward_ctx = LTXForwardContext(
            req=req,
            request_inputs=request_inputs,
            prompt_context=prompt_context,
            device=device,
            guidance_parallel_ready=guidance_parallel_ready,
            attention_kwargs=attention_kwargs,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_mel_bins=latent_mel_bins,
            original_audio_num_frames=original_audio_num_frames,
            padded_audio_num_frames=padded_audio_num_frames,
            timesteps=timesteps_tensor,
            sampler=phase_recipe.sampler,
            audio_scheduler=audio_scheduler,
            video_audio_step_adapter=video_audio_step_adapter,
        )
        video_coords, audio_coords = prepare_rope_coords_stage(pipeline, forward_ctx, latents, audio_latents)
        ring_degree = getattr(pipeline.od_config.parallel_config, "ring_degree", 1) or 1
        if padded_audio_num_frames > original_audio_num_frames and ring_degree > 1:
            raise ValueError(
                "LTX audio padding requires an attention mask, which Ring sequence parallelism does not support. "
                "Use Ulysses-only SP or choose a request whose audio latent length is divisible by the SP size."
            )
        denoise_ctx = LTXDenoiseContext(
            latents=latents,
            audio_latents=audio_latents,
            video_coords=video_coords,
            audio_coords=audio_coords,
            audio_attention_mask=(
                torch.arange(padded_audio_num_frames, device=audio_latents.device)
                .lt(original_audio_num_frames)
                .unsqueeze(0)
                .expand(audio_latents.shape[0], -1)
                if padded_audio_num_frames > original_audio_num_frames
                else None
            ),
            conditioning_mask=conditioning_mask,
        )
        denoise_ctx = pipeline._prepare_denoise_context_for_guidance(forward_ctx, denoise_ctx)
        state = LTXDenoiseExecutor.run(
            pipeline,
            LTXAVState(video=denoise_ctx.latents, audio=denoise_ctx.audio_latents),
            forward_ctx.timesteps,
            lambda index, timestep, state: pipeline._denoise_step(
                index,
                timestep,
                state,
                forward_ctx,
                denoise_ctx,
            ),
        )
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        latents, audio_latents = pipeline._unpack_and_denormalize_stage(
            forward_ctx,
            state.video,
            state.audio,
        )
        normalized_audio = unpack_audio_latents(
            unpad_audio_latents(state.audio, forward_ctx.original_audio_num_frames),
            num_mel_bins=forward_ctx.latent_mel_bins,
        )
        return LTXPhaseResult(
            forward_context=forward_ctx,
            video=latents,
            audio=audio_latents,
            audio_for_next_phase=normalized_audio,
        )
