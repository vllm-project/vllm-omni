# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Official LTX-2 res_2s sampling primitives.

The executor in this module deliberately owns only sampler state. Model
execution and image-conditioning restoration are callbacks, which keeps the
normalized sigma used for every prediction explicit and lets the LTX runtime
compose guidance independently.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import torch

from .ltx2_latents import LTXAVState

LTX_RES2S_STEP_NOISE_SEED = -1
LTX_RES2S_SUBSTEP_NOISE_SEED = 9999
LTX_RES2S_TERMINAL_SIGMA = 0.0011
LTX_RES2S_BONG_ITERATIONS = 100


class LTXRes2sPredictX0(Protocol):
    """Predict denoised video and audio x0 at an explicit normalized sigma."""

    def __call__(self, state: LTXAVState, sigma: torch.Tensor) -> LTXAVState: ...


LTXLatentRestore = Callable[[torch.Tensor], torch.Tensor]
LTXRes2sNoise = Callable[[torch.Tensor, torch.Generator], torch.Tensor]


def build_ltx2_res2s_sigmas(
    num_inference_steps: int,
    video_token_count: int,
    *,
    device: torch.device | str | None = None,
    base_seq_len: int = 1024,
    max_seq_len: int = 4096,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
    terminal: float = 0.1,
) -> torch.Tensor:
    """Build the official token-dependent, terminal-stretched LTX-2 schedule."""
    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least 2 for a terminal-stretched LTX schedule")
    if video_token_count < 1:
        raise ValueError("video_token_count must be positive")
    if max_seq_len == base_seq_len:
        raise ValueError("max_seq_len and base_seq_len must differ")
    if not 0 < terminal < 1:
        raise ValueError("terminal must be strictly between 0 and 1")

    # The official scheduler constructs this schedule on CPU from latent shape
    # metadata and only moves the finished fp32 tensor to the execution device.
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    sigma_shift = video_token_count * slope + (base_shift - slope * base_seq_len)
    exp_shift = math.exp(sigma_shift)
    sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)

    non_zero = sigmas != 0
    one_minus_sigmas = 1.0 - sigmas[non_zero]
    scale = one_minus_sigmas[-1] / (1.0 - terminal)
    sigmas[non_zero] = 1.0 - one_minus_sigmas / scale
    return sigmas.to(dtype=torch.float32, device=device)


def res2s_phi(order: int, negative_h: float) -> float:
    """Evaluate the exponential-integrator phi function used by res_2s."""
    if order < 1:
        raise ValueError("order must be positive")
    if abs(negative_h) < 1e-10:
        return 1.0 / math.factorial(order)
    remainder = sum(negative_h**k / math.factorial(k) for k in range(order))
    return (math.exp(negative_h) - remainder) / negative_h**order


def get_res2s_coefficients(h: float, c2: float = 0.5) -> tuple[float, float, float]:
    """Return the official ``(a21, b1, b2)`` coefficients for one log-sigma step."""
    if c2 == 0:
        raise ValueError("c2 must be non-zero")
    a21 = c2 * res2s_phi(1, -h * c2)
    b2 = res2s_phi(2, -h) / c2
    b1 = res2s_phi(1, -h) - b2
    return a21, b1, b2


def normalized_res2s_noise(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw and normalize the high-precision noise used by the official sampler."""
    dtype = torch.float32 if reference.device.type == "mps" else torch.float64
    noise = torch.randn(reference.shape, generator=generator, dtype=dtype, device=generator.device)
    noise = (noise - noise.mean()) / noise.std()
    return noise.sub_(noise.mean(dim=(-2, -1), keepdim=True)).div_(noise.std(dim=(-2, -1), keepdim=True))


def normalized_res2s_audio_noise(
    reference: torch.Tensor,
    generator: torch.Generator,
    *,
    logical_token_count: int,
) -> torch.Tensor:
    """Draw official noise without letting SP-only padding alter valid audio."""
    if not 0 < logical_token_count <= reference.shape[1]:
        raise ValueError(f"Logical audio token count must be in [1, {reference.shape[1]}], got {logical_token_count}.")
    logical_noise = normalized_res2s_noise(reference[:, :logical_token_count], generator)
    if logical_token_count == reference.shape[1]:
        return logical_noise
    padding = logical_noise.new_zeros(
        reference.shape[0],
        reference.shape[1] - logical_token_count,
        reference.shape[2],
    )
    return torch.cat([logical_noise, padding], dim=1)


def res2s_sde_step(
    sample: torch.Tensor,
    proposed_sample: torch.Tensor,
    sigma: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
    noise: torch.Tensor,
    *,
    eta: float = 0.5,
) -> torch.Tensor:
    """Apply the legacy LTX variance-preserving SDE transition."""
    # The official loop deliberately uses fp64 sigma for the midpoint SDE and
    # the original fp32 schedule values for the full-step SDE. Preserve a
    # tensor caller's dtype; promoting both paths to the sample dtype changes
    # a few bf16 rounding decisions and quickly diverges under stochastic HQ
    # sampling. Python scalars retain the historical sample-dtype behavior.
    sigma = (
        sigma.to(device=sample.device)
        if isinstance(sigma, torch.Tensor)
        else torch.as_tensor(sigma, dtype=sample.dtype, device=sample.device)
    )
    sigma_next = (
        sigma_next.to(device=sample.device)
        if isinstance(sigma_next, torch.Tensor)
        else torch.as_tensor(sigma_next, dtype=sample.dtype, device=sample.device)
    )
    sigma_up = (sigma_next * eta).clamp(max=sigma_next * 0.9999)
    sigma_residual = (sigma_next**2 - sigma_up**2).clamp(min=0).sqrt()
    alpha_ratio = (1 - sigma_next) + sigma_residual
    sigma_down = sigma_residual / alpha_ratio

    if torch.any(sigma_up == 0) or torch.any(sigma_next == 0):
        return proposed_sample

    epsilon = (sample - proposed_sample) / (sigma - sigma_next)
    denoised = sample - sigma * epsilon
    result = alpha_ratio * (denoised + sigma_down * epsilon) + sigma_up * noise
    return result.to(proposed_sample.dtype)


def refine_res2s_anchor(
    midpoint: torch.Tensor,
    denoised_x0: torch.Tensor,
    epsilon: torch.Tensor,
    h: float,
    a21: float,
    *,
    iterations: int = LTX_RES2S_BONG_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed-count official ``bongmath`` anchor recurrence."""
    anchor = denoised_x0 - epsilon
    for _ in range(iterations):
        anchor = midpoint - h * a21 * epsilon
        epsilon = denoised_x0 - anchor
    return anchor, epsilon


def _validate_state(reference: LTXAVState, candidate: LTXAVState) -> None:
    for name in ("video", "audio"):
        expected = getattr(reference, name)
        actual = getattr(candidate, name)
        if actual.shape != expected.shape:
            raise ValueError(f"Predicted {name} x0 shape {actual.shape} does not match latent shape {expected.shape}")
        if actual.device != expected.device:
            raise ValueError(f"Predicted {name} x0 must remain on {expected.device}, got {actual.device}")


def _restore_state(
    state: LTXAVState,
    restore_video: LTXLatentRestore | None,
    restore_audio: LTXLatentRestore | None,
) -> LTXAVState:
    video = state.video if restore_video is None else restore_video(state.video)
    audio = state.audio if restore_audio is None else restore_audio(state.audio)
    if video.shape != state.video.shape or video.device != state.video.device:
        raise ValueError("The video conditioning restore callback must preserve shape and device")
    if audio.shape != state.audio.shape or audio.device != state.audio.device:
        raise ValueError("The audio padding restore callback must preserve shape and device")
    return LTXAVState(video=video, audio=audio)


def _inject_av_sde(
    reference_state: LTXAVState,
    sample: LTXAVState,
    proposed: LTXAVState,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    generator: torch.Generator,
    video_noise_fn: LTXRes2sNoise,
    audio_noise_fn: LTXRes2sNoise,
    eta: float,
    restore_video: LTXLatentRestore | None,
    restore_audio: LTXLatentRestore | None,
) -> LTXAVState:
    # A shared generator and this explicit order match official joint AV sampling.
    video_noise = video_noise_fn(reference_state.video, generator)
    audio_noise = audio_noise_fn(reference_state.audio, generator)
    result = LTXAVState(
        video=res2s_sde_step(sample.video, proposed.video, sigma, sigma_next, video_noise, eta=eta),
        audio=res2s_sde_step(sample.audio, proposed.audio, sigma, sigma_next, audio_noise, eta=eta),
    )
    return _restore_state(result, restore_video, restore_audio)


class LTXRes2sExecutor:
    """Run one LTX AV res_2s phase around an explicit x0 prediction callback."""

    @staticmethod
    def run(  # noqa: PLR0913, PLR0915
        state: LTXAVState,
        sigmas: torch.Tensor,
        predict_x0: LTXRes2sPredictX0,
        *,
        restore_video: LTXLatentRestore | None = None,
        restore_audio: LTXLatentRestore | None = None,
        eta: float = 0.5,
        bongmath: bool = True,
        bongmath_max_iter: int = LTX_RES2S_BONG_ITERATIONS,
        model_dtype: torch.dtype = torch.bfloat16,
        noise_fn: LTXRes2sNoise = normalized_res2s_noise,
        audio_noise_fn: LTXRes2sNoise | None = None,
        on_step: Callable[[], None] | None = None,
    ) -> LTXAVState:
        """Execute res_2s; every invocation creates fresh official phase RNGs."""
        if state.video.device != state.audio.device:
            raise ValueError("Video and audio latents must be on the same device")
        if bongmath_max_iter < 0:
            raise ValueError("bongmath_max_iter must be non-negative")

        device = state.video.device
        sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("sigmas must be a one-dimensional tensor with at least two values")
        if not torch.all(torch.isfinite(sigmas)) or torch.any(sigmas < 0) or torch.any(sigmas > 1):
            raise ValueError("sigmas must contain finite normalized values in [0, 1]")
        if torch.any(sigmas[:-1] <= sigmas[1:]):
            raise ValueError("sigmas must be strictly decreasing")

        num_steps = len(sigmas) - 1
        has_zero_terminal = bool(sigmas[-1] == 0)
        if has_zero_terminal:
            terminal_sigma = sigmas.new_tensor(LTX_RES2S_TERMINAL_SIGMA)
            if bool(sigmas[-2] <= terminal_sigma):
                raise ValueError(
                    "When an LTX Res2s sigma schedule ends at zero, the preceding sigma must be greater than "
                    f"{LTX_RES2S_TERMINAL_SIGMA}."
                )
            sigmas = torch.cat([sigmas[:-1], terminal_sigma.unsqueeze(0), sigmas.new_zeros(1)])

        hp = torch.float32 if device.type == "mps" else torch.float64
        high_precision_sigmas = sigmas.to(hp)
        hs = -torch.log(high_precision_sigmas[1:].cpu() / high_precision_sigmas[:-1].cpu())
        step_generator = torch.Generator(device=device).manual_seed(LTX_RES2S_STEP_NOISE_SEED)
        substep_generator = torch.Generator(device=device).manual_seed(LTX_RES2S_SUBSTEP_NOISE_SEED)
        audio_noise_fn = noise_fn if audio_noise_fn is None else audio_noise_fn

        for step_index in range(num_steps):
            sigma = high_precision_sigmas[step_index]
            sigma_next = high_precision_sigmas[step_index + 1]
            h = hs[step_index].item()
            a21, b1, b2 = get_res2s_coefficients(h)
            midpoint_sigma = torch.sqrt(sigma * sigma_next)

            anchor = LTXAVState(video=state.video.clone().to(hp), audio=state.audio.clone().to(hp))
            denoised_1 = predict_x0(state, sigmas[step_index])
            _validate_state(state, denoised_1)
            denoised_1 = _restore_state(denoised_1, restore_video, restore_audio)
            denoised_1_hp = LTXAVState(video=denoised_1.video.to(hp), audio=denoised_1.audio.to(hp))
            epsilon_1 = LTXAVState(
                video=denoised_1_hp.video - anchor.video,
                audio=denoised_1_hp.audio - anchor.audio,
            )
            midpoint = LTXAVState(
                video=anchor.video + h * a21 * epsilon_1.video,
                audio=anchor.audio + h * a21 * epsilon_1.audio,
            )
            midpoint = _inject_av_sde(
                state,
                anchor,
                midpoint,
                sigma,
                midpoint_sigma,
                substep_generator,
                noise_fn,
                audio_noise_fn,
                0.5,
                restore_video,
                restore_audio,
            )

            if bongmath and h < 0.5 and sigma > 0.03:
                video_anchor, video_epsilon = refine_res2s_anchor(
                    midpoint.video,
                    denoised_1_hp.video,
                    epsilon_1.video,
                    h,
                    a21,
                    iterations=bongmath_max_iter,
                )
                audio_anchor, audio_epsilon = refine_res2s_anchor(
                    midpoint.audio,
                    denoised_1_hp.audio,
                    epsilon_1.audio,
                    h,
                    a21,
                    iterations=bongmath_max_iter,
                )
                anchor = LTXAVState(video=video_anchor, audio=audio_anchor)
                epsilon_1 = LTXAVState(video=video_epsilon, audio=audio_epsilon)

            model_midpoint = LTXAVState(
                video=midpoint.video.to(model_dtype),
                audio=midpoint.audio.to(model_dtype),
            )
            denoised_2 = predict_x0(model_midpoint, midpoint_sigma)
            _validate_state(model_midpoint, denoised_2)
            denoised_2 = _restore_state(denoised_2, restore_video, restore_audio)
            epsilon_2 = LTXAVState(
                video=denoised_2.video.to(hp) - anchor.video,
                audio=denoised_2.audio.to(hp) - anchor.audio,
            )
            proposed = LTXAVState(
                video=anchor.video + h * (b1 * epsilon_1.video + b2 * epsilon_2.video),
                audio=anchor.audio + h * (b1 * epsilon_1.audio + b2 * epsilon_2.audio),
            )
            state = _inject_av_sde(
                state,
                anchor,
                proposed,
                sigmas[step_index],
                sigmas[step_index + 1],
                step_generator,
                noise_fn,
                audio_noise_fn,
                eta,
                restore_video,
                restore_audio,
            )
            state = LTXAVState(video=state.video.to(model_dtype), audio=state.audio.to(model_dtype))
            if on_step is not None:
                on_step()

        if has_zero_terminal:
            denoised = predict_x0(state, sigmas[num_steps])
            _validate_state(state, denoised)
            denoised = _restore_state(denoised, restore_video, restore_audio)
            state = LTXAVState(video=denoised.video.to(model_dtype), audio=denoised.audio.to(model_dtype))
        return state
